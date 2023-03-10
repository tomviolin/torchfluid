Notes:

I was able to get my old GeForce GT 730 card to work as a CUDA GPU by installing pytorch==1.1.0 and
cudatoolkit==9.0.

Only tested in WSL2 Ubuntu 20.04.5 LTS and native Windows with micromamba https://github.com/conda-forge/miniforge#mambaforge

In WSL2 Ubuntu 20.04.5 / micromamba the following sequence got the GT 730 working as cuda device:

```bash
micromamba create -n oldcuda90 pytorch==1.1.0 cudatoolkit==9.0 matplotlib glob2 scipy -c conda-forge -c pytorch -c defaults
```

For native Windows it was much more difficult because pytorch 1.1.0 is no longer available in the conda repositories.

So, I:

* Obtained links to previous versions of pytorch from here: https://pytorch.org/get-started/previous-versions/ 
* Was referred to this page for direct wheel downloads: https://download.pytorch.org/whl/cu90/torch_stable.html
* Downloaded https://download.pytorch.org/whl/cu90/torch-1.1.0-cp37-cp37m-win_amd64.whl
* Established a micromamba environment by:
```ps1
micromamba create -n torch110cu90 numpy -c conda-forge -c pytorch -c defaults
```
Then installed the wheel:
```bash
pip install Downloads\torch-1.1.0-cp37-cp37m-win_amd64.whl
```
then installed the rest of the deps:
```bash
micromamba install cudatoolkit==9.0 matplotlib glob2 scipy
```

It felt like a miracle that I finally tracked everything down.

YMMV.

