import os, shutil
from time import process_time as ptime

import torch
import math

# CUSTOM ESM
#from huggingface_hub import login
from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESM3InferenceClient

# SAMPLING
import sys
sys.path.append('..')
from .ExtendedProtein import ExtendedProtein
from utils.predictions import init_structure_config, predict_structure
from utils.energies import compute_U_cm, compute_U_am, compute_entropy, compute_FoC
from utils.rng_state import load_torch_state, save_torch_state
from utils.operations import compute_q, compute_Hd, mutate, merge_dict, merge_dicts



# ---------------------------- #
# HMC Extended Protein Sampler #
# ---------------------------- #
class ExtendedProteinSampler():

    ## Initialize the sampler
    def __init__(
            self,
            ref_seq: str,
            generator: torch.Generator,
            config_settings: dict,
            name: str = 'ExtendedProteinSampler',
    ):
        self.generator = generator
        self.config_settings = config_settings
        self.config = init_structure_config(**self.config_settings)
        self.name = name

        # Necessary only for the first login
        #login()
        self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open", device=self.generator.device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.ref_eprot = ExtendedProtein(sequence=ref_seq, requires_grad=False, device=self.generator.device)
        self.ref_eprot.expand()
        self.ref_eprot.am = self.model.predict_attention(sequence_probs=self.ref_eprot.psi)
        self.ref_eprot.cm, self.ref_eprot.plddt = predict_structure(
                model=self.model, tokens=self.ref_eprot.tokens, config=self.config, 
                d0=self.config_settings["d0"], infer_cbeta=self.config_settings["infer_cbeta"]
        )

        self._prepare_others()


    ## Perform Hybrid Monte Carlo sampling of the wave function space
    def sample(
            self,
            pars: dict,
            save_settings: dict | None = None,
            start: bool | None = None,
            keep_going: bool = False,
            make_backup: bool = True,
        ):
        eprot, data, pars, save_settings = self._setup(pars, save_settings, start, keep_going)

        t0 = ptime()-data['time']
        eprot_i = eprot.copy()
        for _ in range(pars['moves']):
            move_dt = ptime()

            momenta = self._extract_momenta(eprot.psi, pars)
            K_i = self._compute_K(momenta, pars)
            eprot, momenta, new_data = self._integrate_and_collapse(eprot, momenta, pars)
            K_f = self._compute_K(momenta, pars)
            dE_tot = new_data['U_tot']-data['U_tot'] + K_f-K_i

            is_mutated = (eprot.tokens != eprot_i.tokens).any()
            if is_mutated:
                new_data = merge_dict(
                        self._compute_acceptance_observables(eprot, pars),
                        new_data,
                        overwrite=True,
                )
                dE_cm_seq = new_data['U_cm_seq']-data['U_cm_seq'] + K_f-K_i
                dE_am_seq = new_data['U_am_seq']-data['U_am_seq'] + K_f-K_i
            else:
                dE_cm_seq = K_f-K_i
                dE_am_seq = K_f-K_i

            p = torch.rand(1, device=self.generator.device, generator=self.generator).item()
            if p <= math.exp(-dE_am_seq/pars['T']):
                data = merge_dict(
                        new_data,
                        data,
                        overwrite=True,
                )
                data['am'] += 1
                data['mutations'] = (eprot.tokens != eprot_i.tokens).int().sum().item()
                data['move_dq'] = compute_q(eprot.psi, eprot_i.psi).item()
                eprot_i = eprot.copy()
            else:
                data['mutations'] = 0
                data['move_dq'] = 0.
                eprot = eprot_i.copy()

            data['move'] += 1
            data['time'] = ptime()-t0
            data['move_dt'] = ptime()-move_dt
            data['eff_v'] = math.acos(data['move_dq'])/data['move_dt']
            data['dK'] = K_f-K_i
            data['dE_tot'] = dE_tot
            data['dE_am_seq'] = dE_am_seq
            data['dE_cm_seq'] = dE_cm_seq

            if save_settings["data_step"] > 0 and data["move"]%save_settings["data_step"] == 0: self._save_data(data, f'{save_settings["results_dir"]}/data.dat')
            if save_settings["state_step"] > 0 and data["move"]%save_settings["state_step"] == 0: self._save_state(eprot, save_settings, data["move"])
            if save_settings["print_step"] > 0 and data["move"]%save_settings["print_step"] == 0: self._print_status(data)

        if make_backup:
            self._make_backup(eprot, data)

        return eprot, data


    ## Perform constrained dynamics in the wave function space through velocity Verlet integrator
    def _integrate_and_collapse(
            self,
            eprot: ExtendedProtein,
            momenta: torch.Tensor,
            pars: dict,
    ) -> tuple[ExtendedProtein, torch.Tensor]:
        old_psi = eprot.psi.detach().clone()
        old_Pgrad, _ = self._compute_projected_gradient_and_integration_observables(eprot, pars)

        for istep in range(pars['isteps']):
            with torch.no_grad():
                bad_psi = old_psi + momenta*pars['dt']/pars['m'] - old_Pgrad*pars['dt']**2./(2.*pars['m'])
                scalar_prod = (bad_psi*old_psi).sum(axis=-1)
                Delta = scalar_prod**2. + 1 - (bad_psi**2.).sum(axis=-1)
                assert any(Delta.view(-1) >= 0), f"{self.name}._integrate(): bad integration error has occured. Exiting from the simulation!"
                li_pm = torch.concat(
                    (
                        (-(pars['m']/pars['dt']**2.) * (scalar_prod - torch.sqrt(Delta))).unsqueeze(-1),
                        (-(pars['m']/pars['dt']**2.) * (scalar_prod + torch.sqrt(Delta))).unsqueeze(-1)
                    ),
                    axis=-1
                )
                _, min_index = torch.min(abs(li_pm), dim=-1)
                li = li_pm[torch.arange(min_index.shape[-1]), min_index.view(-1)]
                correction = torch.transpose(li*torch.transpose(old_psi, 0,1), 0,1)
                eprot.psi = bad_psi + correction*pars['dt']**2./pars['m']
            eprot.psi.requires_grad = True
            
            new_Pgrad, new_data = self._compute_projected_gradient_and_integration_observables(eprot, pars)
            with torch.no_grad():
                bad_momenta = momenta - (new_Pgrad+old_Pgrad)*pars['dt']/2. + correction*pars['dt']
                Li = -(1./pars['dt']) * (eprot.psi*bad_momenta).sum(axis=-1)
                Correction = torch.transpose(Li*torch.transpose(eprot.psi, 0,1), 0,1)
                momenta = bad_momenta + Correction*pars['dt']
            
            if istep < pars['isteps']-1:
                old_psi = eprot.psi.detach().clone()
                old_Pgrad = new_Pgrad.detach().clone()

        eprot.collapse()
        return eprot, momenta, new_data
    

    ## Compute gradient projected on the hypersphere for the current wave function (and other integration observables)
    def _compute_projected_gradient_and_integration_observables(self, eprot: ExtendedProtein, pars: dict):
        probs = eprot.psi**2.
        am = self.model.predict_attention(sequence_probs=probs)
        U_am = compute_U_am(am, self.ref_eprot.am)
        entropy = compute_entropy(probs)
        U_tot = pars['slope']*U_am + pars["lambda_S"]*entropy
	
        U_tot.backward(retain_graph=True)
        psi = eprot.psi.detach().clone()
        grad = eprot.psi.grad.detach().clone()
        Pgrad = grad - torch.transpose( (grad*psi).sum(axis=-1)*torch.transpose(psi,0,1), 0,1)

        eprot.psi.grad = None
        eprot.am = am.detach().clone().to("cpu")

        return Pgrad, {"U_tot":U_tot.item(), "U_am":U_am.item(), "entropy":entropy.item()}


    ## Compute observables: potential energy (based on pdb attention matrix) and Hamming distance (with respect to the starting sequence) of the closest sequence
    def _compute_acceptance_observables(self, eprot: ExtendedProtein, pars: dict):
        copy = eprot.copy()
        copy.expand()
        copy.set_requires_grad(False)
        am_seq = self.model.predict_attention(sequence_probs=copy.psi**2.)
        U_am_seq = pars['slope']*compute_U_am(am_seq, self.ref_eprot.am)

        cm, plddt = predict_structure(
                model=self.model, tokens=eprot.tokens, config=self.config,
                d0=self.config_settings["d0"], infer_cbeta=self.config_settings["infer_cbeta"],
        )
        U_cm_seq = compute_U_cm(cm, self.ref_eprot.cm)
        FoC = compute_FoC(cm, self.ref_eprot.cm)
        Hd = compute_Hd(eprot.tokens, self.ref_eprot.tokens)

        eprot.cm = cm.detach().clone().to("cpu")
        eprot.plddt = plddt.detach().clone().to("cpu")

        return {'U_am_seq': U_am_seq.item(), 'U_cm_seq':U_cm_seq.item(), 'FoC': FoC.item(), 'Hd':Hd, 'seq':eprot.sequence}


    ## Compute kinetic energy
    def _compute_K(self, momenta, pars):
        return 1./(2.*pars['m']) * (momenta**2.).sum().item()


    # Extract momenta orthogonal to current wave function psi
    def _extract_momenta(self, psi, pars):
        momenta = torch.randn(*psi.shape, device=self.generator.device, generator=self.generator) * math.sqrt(pars['T']*pars['m'])
        momenta -= torch.transpose( (momenta*psi).sum(axis=-1)*torch.transpose(psi, 0,1), 0,1)
        return momenta.detach().clone()


    ## Save current data to file
    def _save_data(self, data, filename, header=False):
        if header:
            with open(filename, 'w') as f:
                header, line = '', ''
                for key in data:
                    header = header + f'{key}\t'
                    line = line + f'{data[key]}\t'
                print(header[:-1], file=f)
                print(line[:-1], file=f)
        else:
            with open(filename, 'a') as f:
                line = ''
                for key in data: line = line + f'{data[key]}\t'
                print(line[:-1], file=f)


    ## Save current sampler state (eprot and generator)
    def _save_state(self, eprot, save_settings, move):
        eprot.save(dir=save_settings["eprot_dir"], suffix=f'_{move}')
        if move > 0:
            shutil.copy(f'{save_settings["results_dir"]}/generator.npy', f'{save_settings["results_dir"]}/previous_generator.npy')
        save_torch_state(f'{save_settings["results_dir"]}/generator.npy', self.generator)


    ## Print sampled data
    def _print_status(self, data):
        data['time_h'] = data["time"] / 3600.
        data['ar'] = data['am'] / data['move'] if data['move'] > 0 else 0.

        line = ''
        for key, _, fp in self.printing_settings['sampling']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_settings['efficiency']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|'

        data.pop('time_h')
        data.pop('ar')
        print(f'{line}\n{self.separator}')


    ### Make a backup of the current configuration and data to (maybe) restart the simulation
    def _make_backup(self, eprot: ExtendedProtein, data: dict):
        self.backup = {
                'eprot': eprot.copy(),
                'data': data.copy()
        }


    ### Setup new sampling
    def _setup(
            self,
            pars: dict,
            save_settings: dict | None = None,
            start: bool | None = None,
            keep_going: bool = False,
    ):
        pars = self._correct_types(pars, self.types_and_keys['pars'])

        if save_settings is None:
            if self.default_save_settings["eprot_dir"] not in os.listdir():
                os.mkdir(self.default_save_settings["eprot_dir"])
            save_settings = self.default_save_settings.copy()
        elif isinstance(save_settings, dict):
            for default_key in self.default_save_settings.keys():
                if default_key not in save_settings.keys():
                    save_settings[default_key] = self.default_save_settings[default_key]
        else:
            raise TypeError("{self.name}._setup(): save_settings expected types are either Dict or None, but found {type(save_settings)}.")

        if start is None:
            # Start from scratch
            if not keep_going:
                mutant = mutate(self.ref_eprot.sequence, pars['init_muts'], self.generator)
                eprot = ExtendedProtein(sequence=mutant, requires_grad=True, device=self.generator.device)
                eprot.expand()

                data = {
                    'move': 0,
                    'time': 0.,
                }
                _, int_data = self._compute_projected_gradient_and_integration_observables(eprot, pars)
                acc_data = self._compute_acceptance_observables(eprot, pars)
                data = merge_dicts(
                    [
                        int_data,
                        acc_data,
                    ],
                    data
                )
            # Start from the last eprot (ongoing) with a new set of parameters
            else:
                eprot = self.backup['eprot'].copy()
                data = self.backup['data'].copy()
                del self.backup
            data['am'] = 0
            data['mutations'] = 0
            data['move_dq'] = 0.
            data['move_dt'] = 0.
            data['eff_v'] = 0.
            data['dK'] = 0.
            data['dE_tot'] = 0.
            data['dE_am_seq'] = 0.
            data['dE_cm_seq'] = 0.

            if data['move'] == 0:
                if save_settings['data_step'] > 0: self._save_data(data, f'{save_settings["results_dir"]}/data.dat', header=True)
                if save_settings['state_step'] > 0: self._save_state(eprot, save_settings, 0)
        
        # Start from a previous simulation last eprot (stopped)
        else:
            data = start['data'].copy()
            data = self._correct_types(data, self.types_and_keys['data'])
            self.generator = load_torch_state(start['generator'], self.generator)
            eprot = ExtendedProtein(
                    device=self.generator.device,
                    requires_grad=True
            )
            eprot.load(dir=save_settings['eprot_dir'], suffix=f"_{data['move']}")
            assert data['seq'] == eprot.sequence, "{self.name}._setup(): loaded data sequence and eprot.sequence must coincide. Found:\ndata.sequence:{data['seq']}\neprot.sequence:{eprot.sequence}."

        if save_settings['print_step'] > 0:
            print(f'// {self.name} status register:')
            print(f'{self.separator}\n{self.global_header}\n{self.separator}')
            self._print_status(data)

        return eprot, data, pars, save_settings


    ## Convert to correct types each dictionary values
    def _correct_types(self, d, types_and_keys):
        for key in d:
            found = False
            for _type, keys in types_and_keys:
                if key in keys:
                    d[key] = _type(d[key])
                    found = True
                    break
            if not found:
                d[key] = float(d[key])
        return d


    ## Prepare secondary variables of sampler
    def _prepare_others(self):
        self.types_and_keys = {
                'data': [
                    (int, ['move', 'Hd', 'am', 'mutations']),
                    (str, ['seq']),
                ],
                'pars': [
                    (int, ['moves', 'isteps', 'init_muts']),
                ],
        }

        self.default_save_settings = {
                "results_dir": ".",
                "eprot_dir": "./eprot",
                "data_step": 1,
                "state_step": 1000,
                "print_step": 100,
        }

        self.printing_settings = {
                'sampling':[
                    ['move', 'move', 0],
                    ['U_am_seq', 'U_seq', 1],
                    ['U_tot', 'U_tot', 1],
                    ['entropy', 'S', 3],
                    ['mutations', 'mutations', 0],
                    ['Hd', 'Hamming d', 0],
                    ['time_h', 'time', 2],
                ],
                'efficiency':[
                    ['ar', 'ar', 3],
                    ['dE_am_seq', 'dE_seq', 2],
                    ['dE_tot', 'dE_tot', 2],
                ]
        }
        self.separator = ''.join(['-']*(13*len(self.printing_settings['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.printing_settings['efficiency'])+1))
        header = ''
        for _, symbol, _ in self.printing_settings['sampling']: header = f'{header}|{symbol:^12}'
        header = f'{header}|' + ''.join([' ']*5)
        for _, symbol, _ in self.printing_settings['efficiency']: header = f'{header}|{symbol:^12}'
        self.global_header = f'{header}|'
