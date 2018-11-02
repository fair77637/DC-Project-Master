For the imaging team:

1. Useful code is in '/cleancode/' folder
2. The files you want are 'midplanefinderbatch.ipynb' and 'symmetryanalysisbatch.ipynb'
3. For both these files ('midplanefinderbatch.ipynb' and 'symmetryanalysisbatch.ipynb'), all you have to do is put the path of the scan(s) you want into 'paths'. 
4. Run 'midplanefinderbatch.ipynb' to save a midplane binary for a given scan.
5. Run 'symmetryanalysisbatch.ipynb' to get out all the values for the symmetry analysis, and also save all the parenchyma, ventricle and haematoma masks. The code will print out something like: 

                    ['ScanB', 'w1 = 287.118632928', 'w2 = 253.708675649', 'w3 = 66',
                    'Volume of haematoma on LHS is 101216.934341',
                    'Volume of haematoma on RHS is 11828.987111',
                    'Volume of brain on LHS is 736982.079215',
                    'Volume of brain on RHS is 774164.849771',
                    'Volume of CSF in ventricles on LHS is 19655.0864758',
                    'Volume of CSF in ventricles on RHS is 32386.77633'],
                    
  w1, w2 and w3 are the x,y,z coordinates of the ventricle centroid. You can type these coords into ITK snap to get the point.
  These values will be saved in 'info.npy' (which is not a very useful format). You can load this file by typing in :
    import numpy as np
    info =  np.load('info.npy') 
 It is best to copy and paste these values into a spreadsheet or a np array so you can analyse them properly later (and keep    
 track of all your scans).
