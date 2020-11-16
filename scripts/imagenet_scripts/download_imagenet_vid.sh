mkdir ../../data/ &&

wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz \
	-P ../../data/ &&

tar -xvzf ../../data/ILSVRC2015_VID.tar.gz \
	--directory ../../data/ \
	ILSVRC2015_VID/Data/VID &&

rm ../../data/ILSVRC2015_VID.tar.gz
