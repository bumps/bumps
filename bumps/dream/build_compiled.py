import argparse
import multiprocessing
import os
from pathlib import Path
import sysconfig

CC = sysconfig.get_config_vars().get('CC', os.environ.get('CC', 'gcc'))

def compile_dll(nthreads=None, use_openmp=True):
    num_cores = multiprocessing.cpu_count()
    if nthreads == 0:
        nthreads = num_cores

    os.chdir(Path(__file__).parent)
    openmp_flag = "-fopenmp" if use_openmp else ""
    flags = f"-I ../../Random123/include/ -O2 {openmp_flag} -shared -lm -o _compiled.so -fPIC -DMAX_THREADS={nthreads}"
    compile_command = f"{CC} compiled.c {flags}"
    print(f"Compiling: {compile_command}")
    os.system(compile_command)

    return os.path.exists('_compiled.so')

def remove_dll():
    dll_path = Path(__file__).parent / '_compiled.so'
    if dll_path.exists():
        print(f"Removing {dll_path}")
        dll_path.unlink()
    else:
        print(f"{dll_path} does not exist")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nthreads', type=int, default=64, help='Number of threads to compile for, use 0 for all available cores')
    parser.add_argument('--no-openmp', action='store_true', help='Disable OpenMP (e.g. for clang, where it is not supported)')
    parser.add_argument('--remove', action='store_true', help='Remove the compiled dll')
    args = parser.parse_args()
    if args.remove:
        remove_dll()
    else:
        compile_dll(nthreads=args.nthreads, use_openmp=not args.no_openmp)

if __name__ == "__main__":
    main()