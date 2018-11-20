# Imports we need.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import nltk
import nltk.data
import re
import string
from matplotlib import pyplot as plt
from model_CNN_modified import Model


FLAGS = tf.app.flags.FLAGS

# # # tf.app.flags.DEFINE_string("param_name", "default_val", "description")
# tf.app.flags.DEFINE_string("data_path", "/home/tongyu/sentiment_analysis/aclImdb/", "training data dir")
# tf.app.flags.DEFINE_string("train_data_path", '/home/tongyu/sentiment_analysis/aclImdb/train/', "train data path")
# tf.app.flags.DEFINE_string("test_data_path", '/home/tongyu/sentiment_analysis/aclImdb/test/', "test data path")
# tf.app.flags.DEFINE_string("vocab_path", '/home/tongyu/sentiment_analysis/', "vocab path")
# tf.app.flags.DEFINE_string("vocab_file", 'glove.840B.300d.txt', "vocab file")
# tf.app.flags.DEFINE_string("summaries_dir", '/home/tongyu/sentiment_analysis/summaries/', "summaries dir")
# tf.app.flags.DEFINE_string("ckp_dir", '/home/tongyu/sentiment_analysis/ckp_dir/', "check point dir")
# tf.app.flags.DEFINE_string("index_data_path", '/home/tongyu/sentiment_analysis/aclImdb/index_data/', "processed data save path")
#

#
tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_string("data_path", "D:/Imdb/tmp/aclImdb/", "training data dir")
tf.app.flags.DEFINE_string("train_data_path", 'D:/Imdb/tmp/aclImdb/train/', "train data path")
tf.app.flags.DEFINE_string("test_data_path", 'D:/Imdb/tmp/aclImdb/test/', "test data path")
tf.app.flags.DEFINE_string("vocab_path", 'D:/Imdb/tmp/aclImdb/', "vocab path")
tf.app.flags.DEFINE_string("vocab_file", 'glove.840B.300d.txt', "vocab file")
tf.app.flags.DEFINE_string("summaries_dir", 'D:/Imdb/tmp/aclImdb/summaries/', "summaries dir")
tf.app.flags.DEFINE_string("ckp_dir", 'D:/Imdb/tmp/aclImdb/ckp_dir/', "check point dir")
tf.app.flags.DEFINE_string("index_data_path", 'D:/Imdb/tmp/aclImdb/index_data/', "processed data save path")

tf.app.flags.DEFINE_string("mode", 'train', "mode")

#separate：输出一个list,其中每一行是一个经过预处理的句子
#concate：输出一个经过预处理的完整的文本
tf.app.flags.DEFINE_string("output_mode", 'concate', "output_mode")

#定义配置特殊字符
tf.app.flags.DEFINE_string("start_char", '<s>', "start_char")
tf.app.flags.DEFINE_string("end_char", '</s>', "end_char")
tf.app.flags.DEFINE_string("pad_char", '<pad>', "pad_char")
tf.app.flags.DEFINE_string("unk_char", '<unk>', "unk_char")

#max_words定义词表大小
#max_len定义'concate'模式下整个文本长度
#max_seq_len 'separate'模式下每一个子句的长度，与max_len独立
#case_en==1 为对原始输入文本进行大小写替换，除了I以外的所有大小写都会被替换成小写
#padding_en==1 表示根据max_len和max_seq_len用pad_char进行padding
#bn_emb_en==1 表示对预加载的词向量进行归一化，0均值 单位方差
tf.app.flags.DEFINE_integer("max_words", 100000, "max_words")
tf.app.flags.DEFINE_integer("max_len", 500, "max num of tokens per query")
tf.app.flags.DEFINE_integer("case_en", 1, "case_en")
tf.app.flags.DEFINE_integer("max_seq_len", 40, "max_seq_len")
tf.app.flags.DEFINE_integer("padding_en", 1, "padding_en")
tf.app.flags.DEFINE_integer("bn_emb_en", 1, "bn_emb_en")
tf.app.flags.DEFINE_integer("read_limit", 100, "read_limit")
tf.app.flags.DEFINE_float("phi", 0.000001, "phi")


tf.app.flags.DEFINE_string("log_dir", "./logs", " the log dir")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch_size")
tf.app.flags.DEFINE_integer("epoch", 100, "batch_size")
tf.app.flags.DEFINE_integer("emb_dim", 300, "max_words")



pattern = r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |\.\.\.                # ellipsis
              |(?:[.,;"'?():-_`])    # special characters with meanings
            """

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def text_prepro(file_dir,FLAGS):
    case_en = FLAGS.case_en
    output_mode= FLAGS.output_mode
    max_seq_len= FLAGS.max_seq_len
    padding_en= FLAGS.padding_en
    start_char= FLAGS.start_char
    end_char= FLAGS.end_char
    max_len = FLAGS.max_len
    pad_char = FLAGS.pad_char

    with open(file_dir, encoding='utf-8') as f:
        article = []
        text_char = f.read()

        if text_char[-1] in '.!?~)"':
            text_char = text_char[:-1]

        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenizer.tokenize(text_char)
        new_sent = []
        for jj in range(len(sentences)):
            sentence = sentences[jj]
            # words = WordPunctTokenizer().tokenize(sentence)

            words = nltk.regexp_tokenize(sentence, pattern)

            regex_punctuation = re.compile('[%s]' % re.escape(string.punctuation))

            new_words = []
            new_seq = []
            for kk in range(len(words)):
                word = words[kk]
                word_tmp = regex_punctuation.sub("", word)

                if case_en == 1:

                    if word_tmp != '':

                        if word_tmp != 'I':
                            word_tmp = word_tmp.lower()
                        new_words = new_words + [word_tmp]


                else:
                    if word_tmp != '':
                        new_words = new_words + [word_tmp]

            if output_mode == 'separate':

                len_sent = min(len(new_words), max_seq_len - 2)

                if padding_en == 0:
                    sentence = [start_char] + new_words[:len_sent] + [end_char]

                else:
                    len_padding = max_seq_len - len_sent - 2

                    if len_padding > 0:

                        sentence = [start_char] + new_words[:len_sent] + [end_char] + len_padding * [pad_char]
                    else:
                        sentence = [start_char] + new_words[:len_sent] + [end_char]

                article.append(sentence)

            if output_mode == 'concate':
                new_sent = new_sent + new_words

        if output_mode == 'concate':
            len_sent = min(len(new_sent), max_len - 2)

            if padding_en == 0:
                article = [start_char] + new_sent[:len_sent] + [end_char]

            else:
                len_padding = max_len - len_sent - 2

                if len_padding > 0:

                    article = [start_char] + new_sent[:len_sent] + [end_char] + len_padding * [pad_char]
                else:
                    article = [start_char] + new_sent[:len_sent] + [end_char]

    return article



def extract_data(file_name,file_dir,FLAGS):

    data_train = []
    for ii in range(len(file_name)):
        # print(ii)
        file_path = file_dir + file_name[ii]

        article = text_prepro(file_path,FLAGS)

        data_train.append(article)

    return data_train


def text_gen(FLAGS):
    dir_pos = 'pos'
    dir_neg = 'neg'
    dir_unsup = 'unsup'
    read_limit = FLAGS.read_limit
    train_data_path = FLAGS.train_data_path
    test_data_path = FLAGS.test_data_path
    output_mode = FLAGS.output_mode

    pos_file_name = os.listdir(train_data_path + dir_pos)
    neg_file_name = os.listdir(train_data_path + dir_neg)
    uns_file_name = os.listdir(train_data_path + dir_unsup)

    file_dir = train_data_path + dir_pos + '/'
    # pos_file_name = pos_file_name[:read_limit]  # delete
    pos_train = extract_data(pos_file_name, file_dir,FLAGS)
    pos_train_label = np.array(len(pos_file_name) * [[1, 0]])

    file_dir = train_data_path + dir_neg + '/'
    # neg_file_name = neg_file_name[:read_limit]  # delete
    neg_train = extract_data(neg_file_name, file_dir,FLAGS)
    neg_train_label = np.array(len(neg_file_name) * [[0, 1]])

    file_dir = train_data_path + dir_unsup + '/'
    # uns_file_name = uns_file_name[:read_limit]  # delete
    unsup_train = extract_data(uns_file_name, file_dir,FLAGS)

    text_train = pos_train + neg_train
    y_train = np.concatenate([pos_train_label, neg_train_label], axis=0)

    pos_file_name = os.listdir(test_data_path + dir_pos)
    neg_file_name = os.listdir(test_data_path + dir_neg)

    file_dir = test_data_path + dir_pos + '/'
    # pos_file_name = pos_file_name[:read_limit]  # delete
    pos_test = extract_data(pos_file_name, file_dir,FLAGS)
    pos_test_label = np.array(len(pos_file_name) * [[1, 0]])

    file_dir = test_data_path + dir_neg + '/'
    # neg_file_name = neg_file_name[:read_limit]  # delete
    neg_test = extract_data(neg_file_name, file_dir,FLAGS)
    neg_test_label = np.array(len(neg_file_name) * [[0, 1]])

    text_test = pos_test + neg_test
    y_test = np.concatenate([pos_test_label, neg_test_label], axis=0)

    return text_train, y_train, text_test, y_test, unsup_train


def text2token(output_mode,padding_en,text,char2indx):

    seq_indx = []
    char_count = []
    unk_count = []
    for ii in range(len(text)):

        article = text[ii]

        article_char_count = 0
        article_unk_count = 0
        article_indx = []

        if output_mode == 'separate':
            for jj in range(len(article)):

                sentence = article[jj]
                indx = []

                for kk in range(len(sentence)):

                    word = sentence[kk]

                    try:
                        indx = np.append(indx, char2indx[word])

                        if indx[-1] not in [0, 1, 2, 3]:
                            article_char_count = article_char_count + 1

                    except:

                        indx = np.append(indx, 1)
                        article_unk_count = article_unk_count + 1

                indx = np.array(indx, dtype=int)

                article_indx.append(indx)
            char_count.append(article_char_count)
            unk_count.append(article_unk_count)
            article_indx = np.array(article_indx)
            seq_indx.append(article_indx)

        elif output_mode == 'concate':

            sentence = article
            indx = []

            for kk in range(len(sentence)):

                word = sentence[kk]

                try:
                    indx = np.append(indx, char2indx[word])

                    if indx[-1] not in [0, 1, 2, 3]:
                        article_char_count = article_char_count + 1

                except:

                    indx = np.append(indx, 1)
                    article_unk_count = article_unk_count + 1

            indx = np.array(indx, dtype=int)

            # article_indx.append(indx)

            article_indx = indx
            char_count.append(article_char_count)
            unk_count.append(article_unk_count)
            article_indx = np.array(article_indx)
            seq_indx.append(article_indx)

    char_count = np.array(char_count)
    unk_count = np.array(unk_count)

    if padding_en == 1 and output_mode == 'concate':
        seq_indx = np.array(seq_indx)


    return seq_indx,char_count,unk_count



def embedding_prepro(file_dir,FLAGS):
    phi = FLAGS.phi
    bn_emb_en = FLAGS.bn_emb_en
    max_words = FLAGS.max_words
    start_char = FLAGS.start_char
    pad_char = FLAGS.pad_char
    end_char = FLAGS.end_char
    unk_char = FLAGS.unk_char
    with open(file_dir, encoding='utf-8') as f:

        words=[]

        count = 0
        for line in f.readlines():
            sent = line.split()


            if count == 0:
                dim = len(sent[1:])
                embedding = np.zeros([max_words, dim])

            try:

                vec = np.array(sent[1:], dtype=float)
                words.append(sent[0])
            except:
                vec = np.array(sent[-dim:], dtype=float)
                words.append("".join(sent[:-dim]))

            if bn_emb_en == 1:
                embedding[count, :] = (vec-np.mean(vec)) / np.std(vec + phi)
            else:
                embedding[count, :] = vec

            count = count + 1

            if count == max_words:

                if pad_char not in words:
                    extra_vec = np.zeros([1,dim])
                else:
                    indx = words.index(pad_char)
                    extra_vec = embedding[indx][np.newaxis,:]
                    del embedding[indx]
                    del words[indx]

                if unk_char not in words:
                    mu, sigma = 0, 1
                    tmp_vec = np.random.normal(mu, sigma, dim)[np.newaxis,:]
                    extra_vec = np.concatenate([extra_vec,tmp_vec],axis=0)
                else:
                    indx = words.index(pad_char)
                    extra_vec = np.concatenate([extra_vec,embedding[indx][np.newaxis,:]],axis=0)
                    del embedding[indx]
                    del words[indx]

                if start_char not in words:
                    mu, sigma = 0, 1
                    tmp_vec = np.random.normal(mu, sigma, dim)[np.newaxis,:]
                    extra_vec = np.concatenate([extra_vec,tmp_vec],axis=0)
                else:
                    indx = words.index(start_char)
                    extra_vec = np.concatenate([extra_vec,embedding[indx][np.newaxis,:]],axis=0)
                    del embedding[indx]
                    del words[indx]

                if end_char not in words:
                    mu, sigma = 0, 1
                    tmp_vec = np.random.normal(mu, sigma, dim)[np.newaxis,:]
                    extra_vec = np.concatenate([extra_vec,tmp_vec],axis=0)
                else:
                    indx = words.index(end_char)
                    extra_vec = np.concatenate([extra_vec,embedding[indx][np.newaxis,:]],axis=0)
                    del embedding[indx]
                    del words[indx]

                embedding =np.concatenate([extra_vec,embedding],axis=0)

                words = [pad_char,unk_char,start_char,end_char]+words

                if len(words)>max_words:
                    words = words[:max_words]
                    embedding = embedding[:max_words]

                indx2char = dict(zip(list(range(len(words))), words))
                char2indx = dict(zip(words,list(range(len(words)))))
                break
    return [count,dim],indx2char,char2indx,embedding


if __name__ == '__main__':


    if FLAGS.mode == 'prep':

        if os.path.exists(FLAGS.index_data_path) == False:
            os.makedirs(FLAGS.index_data_path)

        if os.path.exists(FLAGS.summaries_dir) == False:
            os.makedirs(FLAGS.summaries_dir)

        if os.path.exists(FLAGS.ckp_dir) == False:
            os.makedirs(FLAGS.ckp_dir)




        print('text generation...')
        text_train, y_train, text_test, y_test, unsup_train = text_gen(FLAGS)

        file_dir = FLAGS.vocab_path+FLAGS.vocab_file

        print('embedding preprocessing...')
        size, indx2char, char2indx, word_mat = embedding_prepro(file_dir,FLAGS)

        print('text to token...')
        x_train,char_count_train,unk_count_train = text2token(FLAGS.output_mode,FLAGS.padding_en,text_train,char2indx)
        x_test, char_count_test, unk_count_test = text2token(FLAGS.output_mode, FLAGS.padding_en, text_test, char2indx)
        x_unsup, char_count_unsup, unk_count_unsup = text2token(FLAGS.output_mode, FLAGS.padding_en, unsup_train, char2indx)

        char_count = np.concatenate([char_count_train,char_count_test,char_count_unsup],axis=0)

        unk_count = np.concatenate([unk_count_train, unk_count_test, unk_count_unsup], axis=0)
        Order = random.sample(range(len(x_train)), len(x_train))

        # imshow = False
        # if imshow == True:
        #     a = unk_count/char_count
        #     x_label = 'unk_ratio'
        #     plt.hist(a, 100)
        #     plt.title('data analysis (10k words)')
        #     plt.xlabel(x_label + ': ave:' + str(int(np.mean(a))))
        #     plt.ylabel('counts')
        #     plt.show()
        #     print(np.mean(a))
        #     print('mean seq len: '+ str(int(np.mean(char_count))))
        #     print('unk ratio：' + str(sum(unk_count)/sum(char_count)))

        print('saving Order...')
        np.save(FLAGS.index_data_path+'Order.npy',Order)
        print('saving x_train...')
        np.save(FLAGS.index_data_path + 'x_train.npy', x_train)
        print('saving y_train...')
        np.save(FLAGS.index_data_path+'y_train.npy',y_train)
        print('saving x_test...')
        np.save(FLAGS.index_data_path+'x_test.npy',x_test)
        print('saving y_test...')
        np.save(FLAGS.index_data_path + 'y_test.npy', y_test)
        print('saving x_unsup...')
        np.save(FLAGS.index_data_path+'x_unsup.npy',x_unsup)
        print('saving word_mat...')
        np.save(FLAGS.index_data_path + 'word_mat.npy', word_mat)


    if FLAGS.mode == 'train':
        print('loading Order...')
        Order = np.load(FLAGS.index_data_path+'Order.npy')
        print('loading x_train...')
        x_train = np.load(FLAGS.index_data_path + 'x_train.npy')
        print('loading y_train...')
        y_train = np.load(FLAGS.index_data_path+'y_train.npy')
        print('loading x_test...')
        x_test = np.load(FLAGS.index_data_path+'x_test.npy')
        print('loading y_test...')
        y_test = np.load(FLAGS.index_data_path + 'y_test.npy')
        print('loading x_unsup...')
        x_unsup = np.load(FLAGS.index_data_path+'x_unsup.npy')
        print('loading word_mat...')
        word_mat = np.load(FLAGS.index_data_path + 'word_mat.npy')

    if FLAGS.mode != 'test':



        print('bilding the model...')
        checkpoint_prefix = os.path.join(FLAGS.ckp_dir, "model")

        g=tf.Graph()

        model = Model(FLAGS, word_mat, graph=g)

        with tf.Session(graph = g) as sess:

            with g.as_default():
                train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, g)
                test_writer = tf.summary.FileWriter(FLAGS.summaries_dir, g)
                init = tf.global_variables_initializer()
                sess.run(init)
                bs = FLAGS.batch_size
                epoch = FLAGS.epoch
                loop =int(len(x_train)/bs)
                saver = tf.train.Saver(max_to_keep=1)

                print('training the model...')
                for i in range(epoch):
                    total_acc = 0
                    total_loss = 0
                    for j in range(loop):

                        batch_xs = x_train[Order[bs*j:bs*j+bs]]
                        batch_ys = y_train[Order[bs*j:bs*j+bs]]


                        _,loss,acc,summary = sess.run([model.train_op,model.loss,model.acc,model.merged],
                                              feed_dict={model.input_x: batch_xs, model.y:batch_ys,
                                                         model.dropout_keep_prob:0.5
                                                         # })
                                                         ,tf.keras.backend.learning_phase(): 1})



                        total_acc += acc
                        total_loss += loss

                        if j % 10==0 and j>0:
                            acc = total_acc/(j+1)
                            loss = total_loss/(j+1)

                            print('epoch: '+str(i)+ '-----acc: '+str(acc)+'----loss: '+str(loss))

                    train_writer.add_summary(summary, i)

                    if i % 20 ==0 and i>0:
                        print('Saving the model')

                        saver.save(sess, FLAGS.ckp_dir+'my-model', global_step=int(i/20))


                    print('validating...')
                    total_acc = 0
                    total_loss = 0
                    for j in range(loop):

                        batch_xs = x_test[Order[bs * j:bs * j + bs]]
                        batch_ys = y_test[Order[bs * j:bs * j + bs]]

                        loss, acc,summary = sess.run([model.loss, model.acc,model.merged],
                                            feed_dict={model.input_x: batch_xs, model.y: batch_ys,
                                                       model.dropout_keep_prob:1
                                                       ,tf.keras.backend.learning_phase(): 0})


                        total_acc += acc
                        total_loss += loss

                        if j == loop-1:
                            acc = total_acc / (j +1)
                            loss = total_loss / (j +1)
                            print('val_acc: ' + str(acc) + '----val_loss: ' + str(loss))
                    test_writer.add_summary(summary, i)

                print('testing....')
                acc=0
                loss=0
                total_acc=0
                total_loss=0
                for j in range(loop):
                    batch_xs = x_test[Order[bs * j:bs * j + bs]]
                    batch_ys = y_test[Order[bs * j:bs * j + bs]]

                    loss, acc = sess.run([model.loss, model.acc],
                                         feed_dict={model.input_x: batch_xs, model.y: batch_ys,
                                                    model.dropout_keep_prob: 1
                                                    , tf.keras.backend.learning_phase(): 0})

                    total_acc += acc
                    total_loss += loss


                acc_fin = total_acc / loop
                loss_fin = total_loss / loop
                print('val_acc: ' + str(acc_fin) + '----val_loss: ' + str(loss_fin))
                print('run: tensorboard --logdir='+FLAGS.summaries_dir)

# tensorboard --logdir D:/Imdb/tmp/aclImdb/summaries/