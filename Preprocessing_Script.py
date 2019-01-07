from nltk.corpus import stopwords

def create_filtered_file(in_file, out_file, stop_words):
    data = []
    
    with open(in_file, 'r', encoding = "utf8") as f:
        content = f.readlines()
    
    for line in content:
        filtered_line = ""
        words = line.split(" ")
        for word in words:
            if not word in stop_words:
                filtered_line += (" " + word)
        data.append(filtered_line)
        
    with open(out_file, 'w', encoding = "utf8") as f:
        f.writelines(data)


#Using standard english stopword filter
stop_words = set(stopwords.words('english'))

training_data = []
test_data = []

create_filtered_file("./amazonreviews/train/ordered_train_pos.txt", "filtered_train_pos.txt", stop_words)
create_filtered_file("./amazonreviews/train/ordered_train_neg.txt", "filtered_train_neg.txt", stop_words)
create_filtered_file("./amazonreviews/test/ordered_test_pos.txt", "filtered_test_pos.txt", stop_words)
create_filtered_file("./amazonreviews/test/ordered_test_neg.txt", "filtered_test_neg.txt", stop_words)




