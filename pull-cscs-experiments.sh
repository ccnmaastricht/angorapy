# zip on server and pull
ssh daint 'zip -r ~/storage.zip /scratch/snx3000/bp000299/dexterous-robot-hand/storage/ -x $SCRATCH/dexterous-robot-hand/storage/experience/*'
scp daint:~/storage.zip .

# unzip locally and move to correct directory
unzip -n storage.zip -d storage/
cp -r storage/scratch/snx3000/bp000299/dexterous-robot-hand/storage/* storage/

# cleanup
rm -r storage/scratch
rm storage.zip