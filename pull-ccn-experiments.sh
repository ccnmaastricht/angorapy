# zip on server and pull
ssh ccn 'tar -czvf ~/storage.tar.gz ~/dexterous-robot-hand/storage/ --exclude=~/dexterous-robot-hand/storage/experience'
scp ccn:~/storage.tar.gz .
ssh ccn 'rm ~/storage.tar.gz'

# unzip locally and move to correct directory
tar -xzvf storage.tar.gz
cp -r home/tonio/dexterous-robot-hand/storage/* storage/

# cleanup
rm -r home/
rm storage.tar.gz