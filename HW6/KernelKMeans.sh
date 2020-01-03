if [ "$#" -lt 3 ] ; then
    echo "$0 <k-cluster> <imagename> {kmeans++|mod|random}"
    exit
fi
rm -f ./kernel-k-means/png/*
rm -f ./kernel-k-means/cluster/*
python3 ./ConvertImage.py "$2"
echo "[shell] convert png to text"
g++ KernelKMeans.cpp -O2 -o KernelKMeans.out
./KernelKMeans.out "$1" "$2" "$3"
echo "[shell] finish cluster"
python3 ./KernelKMeans.py "$1" "$2" "$3"
echo "[shell] generate pictures"
rm -f KernelKMeans.out "$2"