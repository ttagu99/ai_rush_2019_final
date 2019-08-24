
def get_hist_features(cur_article=None, sel_shuffle=False, out_shape=2048):
    if type(cur_article) == str:
        list_article = cur_article.split(',')
    else:
        list_article = []
    
    sel_number=self.hist_maxuse_num
    if sel_number > len(list_article):
        sel_number = len(list_article)

    if sel_shuffle==True:
        sel_article = random.choices(list_article, k=sel_number)
    else:
        sel_article = list_article[:sel_number]
    hist_features = []

    for idx in range(self.hist_maxuse_num):
        if idx<sel_number:
            cnn_feature = self.history_feature_dict[sel_article[idx]]
            hist_features.append(cnn_feature)
        else:
            hist_features.append(np.zeros(out_shape))

    mer_hist_np = np.array(hist_features)
    return mer_hist_np.flatten()
