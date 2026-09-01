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



# ------------------------------- #
# HMC Extended Protein Integrator #
# ------------------------------- #
class ExtendedProteinIntegrator():

    ## Initialize the integrator
    def __init__(
            self,
            ref_seq: str,
            generator: torch.Generator,
            config_settings: dict,
            name: str = 'ExtendedProteinIntegrator',
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
        self.ref_eprot.am = self.model.predict_attention(sequence_probs=self.ref_eprot.psi)#.to("cpu")
        self.ref_eprot.cm, self.ref_eprot.plddt = predict_structure(
                model=self.model, tokens=self.ref_eprot.tokens, config=self.config, 
                d0=self.config_settings["d0"], infer_cbeta=self.config_settings["infer_cbeta"],# to="cpu"
        )

        self._prepare_others()


    ## Perform constrained dynamics in the wave function space through velocity Verlet integrator
    def integrate(
            self,
            pars: dict,
            save_settings: dict | None = None,
    ) -> tuple[ExtendedProtein, torch.Tensor]:
        eprot, momenta, data, pars, save_settings = self._setup(pars, save_settings)
        t0 = ptime()-data['time']

        old_tokens = eprot.tokens.detach().clone()
        old_psi = eprot.psi.detach().clone()
        old_Pgrad, _ = self._compute_projected_gradient_and_integration_observables(eprot, pars)

        for extraction in range(pars['extractions']):
            
            if extraction > 0:
                data['index'] += 1
                data['istep'] = 0
                momenta = self._extract_momenta(eprot.psi, pars)
                data['K'] = self._compute_K(momenta, pars)
                self._save_data(data, f'{save_settings["results_dir"]}/data.dat')
                self._save_state(eprot, save_settings, data)
                self._print_status(data)

            data_i = data.copy()
            
            for istep in range(1, pars['isteps']+1):
                print(f"istep: {istep}")
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

                eprot.collapse()
                is_new = torch.any(old_tokens != eprot.tokens)
                if istep < pars['isteps']-1:
                    old_tokens = eprot.tokens.detach().clone()
                    old_psi = eprot.psi.detach().clone()
                    old_Pgrad = new_Pgrad.detach().clone()

                data['istep'] = istep
                data['time'] = ptime()-t0
                data['K'] = self._compute_K(momenta, pars)
                data = merge_dict(
                        new_data,
                        data,
                        overwrite=True,
                )
                data['dE_tot'] = data['U_tot']-data_i['U_tot'] + data['K']-data_i['K']
                if is_new:
                    data = merge_dict(
                            self._compute_acceptance_observables(eprot, pars),
                            data,
                            overwrite=True,
                    )
                data['dE_am_seq'] = data['U_am_seq']-data_i['U_am_seq'] + data['K']-data_i['K']
                data['dE_cm_seq'] = data['U_cm_seq']-data_i['U_cm_seq'] + data['K']-data_i['K']
                self._save_data(data, f'{save_settings["results_dir"]}/data.dat')
            
                if is_new: 
                    self._save_state(eprot, save_settings, data)
                    self._print_status(data)

        return eprot, data
    

    ## Compute gradient on the current wave function
    def _compute_projected_gradient_and_integration_observables(self, eprot: ExtendedProtein, pars: dict):
        probs = eprot.psi**2.
        am = self.model.predict_attention(sequence_probs=probs)
        U_am = compute_U_am(am, self.ref_eprot.am)
        entropy = compute_entropy(probs, eps=pars["eps"])
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
        FoC = compute_FoC(cm, self.ref_eprot.cm)
        U_cm_seq = compute_U_cm(cm, self.ref_eprot.cm)
        Hd = compute_Hd(eprot.tokens, self.ref_eprot.tokens)

        eprot.cm = cm.detach().clone().to("cpu")
        eprot.plddt = plddt.detach().clone().to("cpu")

        return {'U_am_seq':U_am_seq.item(), 'U_cm_seq':U_cm_seq.item(), 'FoC': FoC.item(), 'Hd':Hd, 'seq':eprot.sequence}


    ## Compute kinetic energy
    def _compute_K(self, momenta, pars):
        return 1./(2.*pars['m']) * (momenta**2.).sum().item()


    # Extract momenta orthogonal to current wave function psi
    def _extract_momenta(self, psi, pars):
        momenta = torch.randn(*psi.shape, device=self.generator.device, generator=self.generator) * math.sqrt(pars['T']*pars['m'])
        momenta -= torch.transpose( (momenta*psi).sum(axis=-1)*torch.transpose(psi,0,1), 0,1)
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
    def _save_state(self, eprot, save_settings, data):
        eprot.save(dir=save_settings["eprot_dir"], suffix=f'_{data["index"]}_{data["istep"]}')
        if data['index']+data['istep'] > 0:
            shutil.copy(f'{save_settings["results_dir"]}/generator.npy', f'{save_settings["results_dir"]}/previous_generator.npy')
        save_torch_state(f'{save_settings["results_dir"]}/generator.npy', self.generator)


    ## Print sampled data
    def _print_status(self, data):
        data['time_h'] = data["time"] / 3600.

        line = ''
        for key, _, fp in self.printing_settings['sampling']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_settings['efficiency']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|'

        data.pop('time_h')
        print(f'{line}\n{self.separator}')


    ### Setup new sampling
    def _setup(
            self,
            pars: dict,
            save_settings: dict | None = None,
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
        save_settings['state_step'] = math.floor(save_settings['state_step']/save_settings['data_step']) * save_settings['data_step']

        #eprot = self.ref_eprot.copy()
        mutant = mutate(self.ref_eprot.sequence, pars['init_muts'], self.generator)
        eprot = ExtendedProtein(sequence=mutant, requires_grad=True, device=self.generator.device)
        eprot.expand()
        momenta = self._extract_momenta(eprot.psi, pars)
        
        data = {
                'index': 0,
                'istep': 0,
                'time': 0.,
                'mutations': 0,
                'dE_tot': 0.,
                'dE_am_seq': 0.,
                'dE_cm_seq': 0.,
                'K': self._compute_K(momenta, pars),
        }
        _, int_data = self._compute_projected_gradient_and_integration_observables(eprot, pars)
        acc_data = self._compute_acceptance_observables(eprot, pars)
        data = merge_dicts(
                [
                    int_data,
                    acc_data,
                ],
                data,
                overwrite=True,
        )

        self._save_data(data, f'{save_settings["results_dir"]}/data.dat', header=True)
        self._save_state(eprot, save_settings, data)
        print(f'// {self.name} status register:')
        print(f'{self.separator}\n{self.global_header}\n{self.separator}')
        self._print_status(data)

        return eprot, momenta, data, pars, save_settings


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
                    (int, ['index', 'istep', 'Hd']),
                    (str, ['seq']),
                ],
                'pars': [
                    (int, ['isteps', 'extractions', 'init_muts']),
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
                    ['index', 'index', 0],
                    ['istep', 'istep', 0],
                    ['U_am_seq', 'U_seq', 1],
                    ['U_tot', 'U_tot', 1],
                    ['entropy', 'S', 1],
                    ['Hd', 'Hamming d', 0],
                    ['time_h', 'time', 2],
                ],
                'efficiency':[
                    ['dE_am_seq', 'dE_seq', 3],
                    ['dE_tot', 'dE_tot', 3],
                ]
        }
        self.separator = ''.join(['-']*(13*len(self.printing_settings['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.printing_settings['efficiency'])+1))
        header = ''
        for _, symbol, _ in self.printing_settings['sampling']: header = f'{header}|{symbol:^12}'
        header = f'{header}|' + ''.join([' ']*5)
        for _, symbol, _ in self.printing_settings['efficiency']: header = f'{header}|{symbol:^12}'
        self.global_header = f'{header}|'
