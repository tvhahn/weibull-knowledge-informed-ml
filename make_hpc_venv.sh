#!/bin/bash
cd
module load python/3.8
virtualenv ~/weibull
source ~/weibull/bin/activate
pip install --no-index --upgrade pip
pip install --no-index pandas scipy scikit_learn matplotlib seaborn
pip install --no-index torch jupyterlab click
pip install python-dotenv

# create bash script for opening jupyter notebooks https://stackoverflow.com/a/4879146/9214620
cat << EOF >$VIRTUAL_ENV/bin/notebook.sh
#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter-lab --ip \$(hostname -f) --no-browser
EOF

chmod u+x $VIRTUAL_ENV/bin/notebook.sh

# install unrar in environment since it does not
# exist by default on HPC system -- make from source
cd scratch
wget https://www.rarlab.com/rar/unrarsrc-6.0.7.tar.gz
tar -xf unrarsrc-6.0.7.tar.gz
cd unrar
make -f makefile
install -v -m755 unrar ~/weibull/bin