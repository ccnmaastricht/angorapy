# zip on server and pull
ssh office-pc 'zip -r ~/storage.zip ~/workspace/dexterous-robot-hand/storage/ -x ~/workspace/dexterous-robot-hand/storage/experience/*'
scp office-pc:~/storage.zip .
ssh office-pc 'rm ~/storage.zip'

# unzip locally and move to correct directory
unzip storage.zip
cp -r home/weidler/workspace/dexterous-robot-hand/storage/* storage/

# cleanup
rm -r home/
rm storage.zip