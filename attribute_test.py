import pickle


if __name__ == "__main__":
    data = pickle.load(open('data/annotations/annotations.pkl', 'rb'))

    images = data['images']
    attributes = data['attributes']
    categories = data['categories']
    labels = data['labels']
    assert len(labels) == len(images)
    print(attributes)