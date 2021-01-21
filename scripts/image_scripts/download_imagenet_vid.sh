mkdir -p ../../data/ &&

wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz \
	-P ../../data/ &&

tar -xvzf ../../data/ILSVRC2015_VID.tar.gz \
	--directory ../../data/ &&

rm ../../data/ILSVRC2015_VID.tar.gz
