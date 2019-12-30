I use the OIDv4 toolkit (https://github.com/sarmatkasaev/OIDv4_ToolKit) to download the images I need and convert them to detectron format

#### Project folders
* gloves (gloves detection)

#### Dataset structure
```
Dataset
|
|   class-descriptions-boxable.csv
|   train-annotations-bbox.csv
|   validation-annotations-bbox.csv
|   test-annotations-bbox.csv
|
└─── test
|
└─── train
|
└─── validation
     |
     └───Apple
     |     |
     |     |0fdea8a716155a8e.jpg
     |     |2fe4f21e409f0a56.jpg
     |     |...
     |     └───Labels
     |            |
     |            |0fdea8a716155a8e.txt
     |            |2fe4f21e409f0a56.txt
     |            |...
     |
     └───Orange
           |
           |0b6f22bf3b586889.jpg
           |0baea327f06f8afb.jpg
           |...
           └───Labels
                  |
                  |0b6f22bf3b586889.txt
                  |0baea327f06f8afb.txt
                  |...
```