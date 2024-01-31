text = open("data_drones/RGB_Augmented/Train_Labels_CSV.csv")
text = ''.join([i for i in text]) \
    .replace("images/", "")
    
text = ''.join([i for i in text]) \
    .replace("stressed", "2")
text = ''.join([i for i in text]) \
    .replace("healthy", "1")

x = open("New_Train_Labels_RGB.csv","w")
x.writelines(text)
x.close()

text = open("data_drones/RGB_Augmented/Test_Labels_CSV.csv")
text = ''.join([i for i in text]) \
    .replace("images/", "")

text = ''.join([i for i in text]) \
    .replace("stressed", "2")
text = ''.join([i for i in text]) \
    .replace("healthy", "1")

x = open("New_Test_Labels_RGB.csv","w")
x.writelines(text)
x.close()