text = open("data_drones/Spectral_Augmented/Labels/Train_Labels_CSV.csv")
text = ''.join([i for i in text]) \
    .replace("images/image", "Image_")

for i in range(10):
    text = ''.join([i for i in text]) \
        .replace("Image_"+str(i)+".jpg", "Image_0"+str(i)+".jpg")
    
text = ''.join([i for i in text]) \
    .replace("stressed", "2")
text = ''.join([i for i in text]) \
    .replace("healthy", "1")

x = open("New_Train_Labels.csv","w")
x.writelines(text)
x.close()

text = open("data_drones/Spectral_Augmented/Labels/Test_Labels_CSV.csv")
text = ''.join([i for i in text]) \
    .replace("images/Image_", "Image_")

for i in range(10):
    text = ''.join([i for i in text]) \
        .replace("Image_"+str(i)+".jpg", "Image_0"+str(i)+".jpg")

text = ''.join([i for i in text]) \
    .replace("stressed", "2")
text = ''.join([i for i in text]) \
    .replace("healthy", "1")

x = open("New_Test_Labels.csv","w")
x.writelines(text)
x.close()