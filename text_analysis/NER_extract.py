import nltk

def extract_entity_names(t):
    #entity_names = []
    count = 0
    if hasattr(t, 'node') and t.node:
        if t.node == 'NE':
            #entity_names.append(' '.join([child[0] for child in t]))
            count += 1
        else:
            for child in t:
                #entity_names.extend(extract_entity_names(child))
                count += extract_entity_names(child)
                
    #return entity_names
    return count
 
#entity_names = []

def NERCount(sample):
    #sample = 'Born into an aristocratic Bengali family of Calcutta'
#sample = "I am Jhon rom America"
    sentences = nltk.word_tokenize(sample)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.batch_ne_chunk(tagged_sentences, binary=True)
    count = 0
    for tree in chunked_sentences:
    # Print results per sentence
    # print extract_entity_names(tree)
    
    #entity_names.extend(extract_entity_names(tree))
        count += extract_entity_names(tree)
        return count
 
# Print all entity names
#print entity_names
 
# Print unique entity names
#print list(set(entity_names))
#print count 