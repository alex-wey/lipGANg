import glob

def win():
    # loop through all of trainvals folders and get the folder names
    # get the file names too
    final = set()
    final_list = []
    # TODO: note, we moved trainval to the lipGANg!!!
    l = glob.glob("/Users/ethanchung/Programming/cs147/final_project/LipGAN-fully_pythonic/lipGANg/trainval/*")
    for folder in l:
        x = glob.glob(folder + "/*")
        for items in x:
            list_x = items.split("/")
            list_x = list_x[-2:]
            list_x[-1] = list_x[-1][:-4]
            joined = list_x[-2] + "/" + list_x[-1] + "\n"
            #print(joined)
            if joined not in final:
                final.add(joined)
                final_list.append(joined)

    
    L = final_list
    
    # write to file
    file1 = open('filelists/train_val.txt', 'w')
    file1.writelines(L) 

    file1.close()

win()