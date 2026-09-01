import torch
import numpy as np
import pandas as pd
import os


# Read desired inputs from <f> file
def load_inputs(f, start=None, end=None):
    with open(f, 'r') as ff:
        lines = ff.readlines()

    if (start is not None) and (end is not None):
        if isinstance(start, int) and isinstance(end, int):
            lines = lines[start:end]
        elif isinstance(start, str) and isinstance(end, str):
            pos_start = 0
            while lines[pos_start][:len(start)] != start and pos_start < len(lines)-1:
                pos_start += 1
            pos_end = pos_start+1
            while lines[pos_end][:len(end)] != end and pos_end < len(lines)-1:
                pos_end += 1
            if pos_end == len(lines)-1: pos_end = None
            lines = lines[pos_start:pos_end]
        else:
            raise TypeError(f"load_inputs(): start and end should be of the same type, either string or integer, but found {type(start)} and {type(end)} respectively.")

    elif start is not None:
        if isinstance(start, int):
            lines = lines[start:]
        elif isinstance(start, str):
            pos_start = 0
            while lines[pos_start][:len(start)] != start and pos_start < len(lines)-1:
                pos_start += 1
            lines = lines[pos_start:]
        else:
            raise TypeError(f"load_inputs(): start should be either a string or an integer, but found {type(start)}.")

    elif end is not None:
        if isinstance(end, int):
            lines = lines[:end]
        elif isinstance(end, str):
            pos_end = 0
            while lines[pos_end][:len(end)] != end and pos_end < len(lines)-1:
                pos_end += 1
            if pos_end == len(lines)-1: pos_end = None
            lines = lines[:pos_end]
        else:
            raise TypeError(f"load_inputs(): end should be either a string or an integer, but found {type(end)}.")

    inputs = {}
    for line in lines:
        skip_line_conditions = [
            line == '\n',
            line.lstrip('\t')[0] == '#',
        ]
        if any(skip_line_conditions): continue
        comment_idx, spaces_idx = 0, 0
        while line[comment_idx] != '#' and comment_idx < len(line)-1:
            spaces_idx = spaces_idx if line[comment_idx] in [' ', '\t'] else comment_idx
            comment_idx += 1
        line = line[:spaces_idx+1]
        key_end_idx, value_start_idx = None, None
        idx = 0
        while value_start_idx is None:
            if key_end_idx is None:
                if line[idx] == ' ': key_end_idx = idx
            else:
                if (value_start_idx is None) and (line[idx] != ' '): value_start_idx = idx
            idx += 1
        key, value = line[0:key_end_idx], line[value_start_idx:]
        value = value[:-1] if value[-1]=='\n' else value
        if (value[0] == '[') and (value[-1] == ']'):
            values_list = value[1:-1].split(',')
            try: inputs[key] = [eval(el) for el in values_list]
            except: inputs[key] = [el.replace(' ', '') for el in values_list]
        else:
            try: inputs[key] = eval(value)
            except: inputs[key] = value
    return inputs


# Create path
def create_path(path):
    dirs = path.split('/')
    if dirs[0] == '':
        dirs.pop(0)
        dirs[0] = f"/{dirs[0]}"
    elif set(dirs[0]) != {'.'}:
        dirs = ['./'] + dirs

    actual_dir = dirs.pop(0)
    for idx, new_dir in enumerate(dirs):
        cwd_dirlist = [d for d in os.listdir(f'{actual_dir}') if os.path.isdir(f'{actual_dir}/{d}')]
        if not new_dir in cwd_dirlist:
            os.mkdir(f'{actual_dir}/{new_dir}')
        actual_dir = f'{actual_dir}/{new_dir}'


# Find correct path based on pars file
def find_path(raw_path, dname, pfile, pname, lpfunc, **kwargs):
    pars = lpfunc(pfile, **kwargs)

    dirlist = [d for d in os.listdir(raw_path) if os.path.isdir(f"{raw_path}/{d}") and (dname in d)]

    for d in dirlist:
        saved_pars = lpfunc(f"{raw_path}/{d}/{pname}", **kwargs)
        if pars == saved_pars:
            return f"{raw_path}/{d}"

    counter_check = any([f=='counter.txt' for f in os.listdir(raw_path) if os.path.isfile(f"{raw_path}/{f}")])
    if not counter_check:
        counter = 0
    else:
        with open(f"{raw_path}/counter.txt", 'r') as f:
            lines = f.readlines()
        assert len(lines)==1, f'find_path(): Too many lines found in "counter.txt" file in {raw_path} directory. Expected: 1, found: {len(lines)}.'
        counter = int(lines[0])
    path = f"{raw_path}/{dname}{counter}"
    create_path(path)
    with open(f"{raw_path}/counter.txt", 'w') as f:
        print(f'{counter+1}', file=f)

    with open(f"{pfile}", "r") as f:
        plines = f.readlines()
    with open(f"{path}/{pname}", "w") as f:
        for pline in plines:
            print(pline, file=f, end='')
    return path


# Clean path (and whatever is within it)
def clean_path(path, skip=[]):
    for el in os.listdir(path):
        if os.path.isfile(f"{path}/{el}"):
            if el in skip: continue
            os.remove(f"{path}/{el}")
        else:
            clean_path(el)
            os.rmdir(el)


# Load sampling data or check dictionary
def load_stuff(f, heavy=False, step=10, sep='\t'):
    if heavy:
        stuff = load_heavy_csv(f, step, sep)
    else:
        with open(f, 'r') as ff:
            stuff = pd.read_csv(ff, sep=sep, header=0)
    return stuff


# Load heavy csv file
def load_heavy_csv(f, step=10, sep='\t'):
    assert step > 1, f'load_heavy_csv(): variable step must be greater than 1 (found {step}). Exit!'
    ff = open(f, 'r')

    header = ff.readline()
    header = header.split(sep)
    header[-1] = header[-1][:-1] #eliminate the "\n" in the last column name

    iline = 0
    lines = []
    while True:
        line = ff.readline()
        if not line: break
        elif iline%step == 0:
            line = [float(value) for value in line.split(sep) if (value != '\n')]
            lines.append(line)
        iline += 1

    data = pd.DataFrame(lines, columns=header)
    return data


# Return the corresponding move from a simulation file
def get_move(f, prefix):
    return int(f.split('/')[-1].replace(prefix, '').replace('.pt', ''))


# Load files (of training or monte carlo simulations) from psi directory
def get_files(dir, prefix, files_number=None, add_start=True, min_move=None, max_move=None, return_moves=False):
    lp = len(prefix)

    try:
        files = sorted(
            [f'{dir}/{f}' for f in os.listdir(dir)
            if os.path.isfile(f'{dir}/{f}') and (f != f'{prefix}0.pt') and (f[:lp] == prefix)],
            key=lambda f: get_move(f, prefix)
        )
    except:
        files = sorted(
            [f'{dir}/{f}' for f in os.listdir(dir)
            if os.path.isfile(f'{dir}/{f}') and (f != f'{prefix}0.pt') and (f[:lp] == prefix) and (f != f'{prefix}f.pt')],
            key=lambda f: get_move(f, prefix)
        )

    if min_move is not None:
        files = [f for f in files if get_move(f, prefix) >= min_move]
    if max_move is not None:
        files = [f for f in files if get_move(f, prefix) <= max_move]
    if add_start and (f'{prefix}0.pt' in os.listdir(dir)):
        files = [f'{dir}/{prefix}0.pt'] + files
    if files_number is not None:
        indexes = np.unique(np.linspace(0, len(files)-1, files_number).astype(int))
        files = np.array(files)[::-1]
        files = files[indexes]
        files = list(files)[::-1]
    files = np.array(files)

    if return_moves:
        moves = np.array([get_move(f, prefix) for f in files])
        return files, moves
    else:
        return files


# Read sequence, distance matrix (and compute correspondent attention matrix)
def read_from_name(seq_name, pdb_dir, return_attmatrix=False):
    with open(f'{pdb_dir}/{seq_name}_seq.txt', 'r') as f:
        sequence = f.readline()[:-1]

    with open(f'{pdb_dir}/{seq_name}_distmatrix.pt', 'rb') as f:
        distmatrix = torch.load(f)
    distmatrix = distmatrix.unsqueeze(0)

    if not return_attmatrix:
        return sequence, distmatrix
    else:
        attmatrix = 1.-distmatrix/distmatrix.max()
        attmatrix = attmatrix - torch.eye(len(sequence)).unsqueeze(0)
        for i in range(len(sequence)-1):
            attmatrix[0][i, i+1] = 0
            attmatrix[0][i+1, i] = 0
        return sequence, distmatrix, attmatrix
