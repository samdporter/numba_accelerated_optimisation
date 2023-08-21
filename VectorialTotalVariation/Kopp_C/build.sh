# remove previously build files
rm _zheevh.so
rm zheevh3_wrap.c
rm zheevh3_wrap.o

# file: build.sh
swig -python -Isrc zheevh3.i

gcc -Isrc -fPIC $(pkg-config --cflags --libs python3) -c zheevh3.c zheevh3_wrap.c
gcc -shared -fPIC -o _zheevh3.so zheevh3.o zheevh3_wrap.o

python -c "import _zheevh3" # test