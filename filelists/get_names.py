import glob

def win():
    # loop through all of trainvals folders and get the folder names
    # get the file names too
    final = set()
    l = glob.glob("/Users/ethanchung/Programming/cs147/final_project/LipGAN-fully_pythonic/trainval/*")
    for folder in l:
        x = glob.glob(folder + "/*")
        for items in x:
            list_x = items.split("/")
            list_x = list_x[-2:]
            list_x[-1] = list_x[-1][:-4]
            joined = list_x[-2] + "/" + list_x[-1] + "\n"
            #print(joined)
            final.add(joined)
    
    L = list(final)
    
    # write to file
    file1 = open('filelists/train_val.txt', 'w')
    file1.writelines(L) 

    file1.close()

win()