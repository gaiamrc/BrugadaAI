import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance, plot_tree
import optuna
import warnings


warnings.filterwarnings("ignore")
np.random.seed(123)



# -------------------------------------------------------------------------------------
# Functions definition
# -------------------------------------------------------------------------------------
def model_preprocessing(dataframe):

    dataframe = dataframe.drop(labels='ECG utilizzato', axis=1)
    # dataframe = dataframe.drop(labels='Sex', axis=1)
    # dataframe = dataframe.drop(labels='Age', axis=1)

    sex_le = LabelEncoder()
    sex_label = sex_le.fit_transform(dataframe['Sex'])
    dataframe['Sex'] = sex_label

    dataframe['Age'].fillna(dataframe['Age'].mode()[0], inplace=True)
    dataframe['PRint'].fillna((dataframe['PRint'].mode()[0]), inplace=True)
    dataframe['QRSaxis'].fillna((dataframe['QRSaxis'].mode()[0]), inplace=True)
    dataframe['QRSwidDII'].fillna((dataframe['QRSwidDII'].median()), inplace=True)
    dataframe['QRSwidV6'].fillna((dataframe['QRSwidV6'].median()), inplace=True)
    dataframe['QTintV5'].fillna((dataframe['QTintV5'].median()), inplace=True)
    dataframe['cQTintV5'].fillna((dataframe['cQTintV5'].median()), inplace=True)
    dataframe['TpeakTendV2'].fillna((dataframe['TpeakTendV2'].median()), inplace=True)   
    dataframe['JelevV1'].fillna((dataframe['JelevV1'].median()), inplace=True)
    dataframe['JelevV2'].fillna((dataframe['JelevV2'].median()), inplace=True)
    dataframe['SwidDI'].fillna((dataframe['SwidDI'].median()), inplace=True)
    dataframe['SlenDI'].fillna((dataframe['SlenDI'].median()), inplace=True)
    dataframe['RwidaVR'].fillna((dataframe['RwidaVR'].mode()[0]), inplace=True)
    dataframe['RlenaVR'].fillna((dataframe['RwidaVR'].mode()[0]), inplace=True)
    dataframe['Type1aVR'] = dataframe['Type1limb'].fillna(0.5)
    dataframe['fQRS'] = dataframe['fQRS'].fillna(0.5)

    dataframe = dataframe.dropna()

    return dataframe


def correlation_blocks(dataframe, name='correlation_blocks.pdf', prefix=''):
    # ls1 = ['FC','PRint','QRSaxis','QRSwidDII','QRSwidV1','QRSwidV2','QRSwidV6']
    ls1 = ['Sex','Age','FC','PRint','QRSaxis','QRSwidDII','QRSwidV1','QRSwidV2','QRSwidV6']
    ls2 = ['QTintV5','QTintV2','cQTintV5','cQTintV2','TpeakTendV2','JTendV2']
    ls3 = ['JelevV1','JelevV2','SwidDI','SlenDI','RwidaVR','RlenaVR','Type1limb','fQRS']

    dataframe1 = dataframe[ls1].copy()
    dataframe2 = dataframe[ls2].copy()
    dataframe3 = dataframe[ls3].copy()

    corr1 = dataframe1.corr()
    corr2 = dataframe2.corr()
    corr3 = dataframe3.corr()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))
    # ax1.title.set_text('FC, PR int, QRS axis, QRS width')
    ax1.title.set_text('Sex, Age, FC, PR int, QRS axis, QRS width')
    ax2.title.set_text('QT int, Tpeak-Tend, J-Tend')
    ax3.title.set_text('J height, S wave, R wave, Type1 presence, fragmentation presence')

    sns.heatmap(corr1, annot=True, ax=ax1)
    sns.heatmap(corr2, annot=True, ax=ax2)
    sns.heatmap(corr3, annot=True, ax=ax3)

    fig.tight_layout()
    fig.savefig(f'../figs/{prefix}{name}')


def column_hist(a, col_sig, col_bkg, nbins, log=False):
    SIG_COL = ('#F55E5A')
    BKG_COL = ('#17B3B7')

    # min_feat = col_sig.min() if col_sig.min() <= col_bkg.min() else col_bkg.min()
    # max_feat = col_sig.max() if col_sig.max() >= col_bkg.max() else col_bkg.max()
    # print(min_feat, max_feat)

    # sig = col_sig[(col_sig > xlim[0]-5e-3) & (col_sig < xlim[1]+5e-3)]
    # bkg = col_bkg[(col_bkg > xlim[0]-5e-3) & (col_bkg < xlim[1]+5e-3)]

    a.hist(col_sig, weights=np.ones(len(col_sig)) / len(col_sig), bins=nbins, log=log, label='event', alpha=0.7, color=SIG_COL, antialiased=True, histtype = 'stepfilled')
    a.hist(col_bkg, weights=np.ones(len(col_bkg)) / len(col_bkg), bins=nbins, log=log, label='control', alpha=0.7, color=BKG_COL, antialiased=True, histtype = 'stepfilled')


def plot_column_hist(dataframe, dataframe1, dataframe0, prefix=''):
    # titles = ['FC','PR int','QRS axis','QRS wid DII','QRS wid V1','QRS wid V2','QRS wid V6','QT int V5','QT int V2','cQT int V5','cQT int V2','Tpeak-Tend V2','J-TendV2','J height V1','J height V2','S wid DI','S len DI','R wid aVR','R len aVR','Type1 aVR','Type1 other limb lead','fQRS']
    titles = ['Sex','Age','FC','PR int','QRS axis','QRS wid DII','QRS wid V1','QRS wid V2','QRS wid V6','QT int V5','QT int V2','cQT int V5','cQT int V2','Tpeak-Tend V2','J-TendV2','J height V1','J height V2','S wid DI','S len DI','R wid aVR','R len aVR','Type1 in peripheral lead','fQRS']

    fig, axs = plt.subplots(6, 4, figsize=(15, 20))

    for ax, tit, feat in zip(axs.flatten(), titles, dataframe.columns[0:23]):
        ax.title.set_text(tit)
        column_hist(ax, dataframe1[feat], dataframe0[feat], 20)
    
    fig.tight_layout()
    
    fig.savefig(f'../figs/{prefix}col_histogram.pdf')

        
def objective(trial):
    max_depth = trial.suggest_int('max_depth', 1, 20)
    n_estimators = trial.suggest_int('n_estimators', 2, 50)
    # eta =  trial.suggest_loguniform('eta', 1e-8, 1.0)
    # gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
    # min_child_weight = trial.suggest_loguniform('min_child_weight', 1e-8, 1.0)
    # max_delta_step = trial.suggest_loguniform('max_delta_step', 1e-8, 10)
    # subsample = trial.suggest_uniform('subsample', 1e-8, 1.0)
    # reg_lambda = trial.suggest_uniform('reg_lambda', 0.0, 1000.0)
    # reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 1000.0)
    
    model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, tree_method='exact', objective='binary:logistic')
    # model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, eta=eta, gamma=gamma, min_child_weight=min_child_weight, max_delta_step=max_delta_step,
    #                        subsample=subsample, tree_method='exact', objective='binary:logistic')
 
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    f1_mean = score.mean()

    return f1_mean




def model_metrics(classifier, estimator, train_features, train_labels, test_features, test_labels, pred_labels_test, pred_labels_train, prefix='', suffix=''):
    
    # Basic info about the model
    print('*************** Model Summary ***************')
    print('No. of classes: ', classifier.n_classes_)
    print('Classes: ', classifier.classes_)
    print('No. of features: ', classifier.n_features_in_)
    print('Feature importance: ', classifier.feature_importances_)

    fig, ax = plt.subplots()
    plot_importance(estimator, ax, importance_type='gain')
    fig.savefig(f'../figs/{prefix}feat_imp{suffix}.pdf')

    fig, ax = plt.subplots()
    plot_tree(estimator, ax=ax)
    fig.savefig(f'../figs/{prefix}tree{suffix}.pdf')

    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Test Data ***************')
    score_te = estimator.score(test_features, test_labels)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(test_labels, pred_labels_test))
    
    cmd = plot_confusion_matrix(estimator, test_features, pred_labels_test, labels=[1,0], cmap='GnBu')  
    fig = cmd.figure_
    fig.savefig(f'../figs/{prefix}test_conf_mtx{suffix}.pdf')


    ### ROC-AUC + precision-recall
    ns_probs_te = [0 for _ in range(len(test_labels))]
    clf_probs_te = estimator.predict_proba(test_features)
    clf_probs_te = clf_probs_te[:, 1]
    
    # calculate scores
    ns_rauc_te = roc_auc_score(test_labels, ns_probs_te)
    clf_rauc_te = roc_auc_score(test_labels, clf_probs_te)
    # summarize scores
    print('ROC AUC=%.3f' % (clf_rauc_te))
    # calculate roc curves
    ns_fpr_te, ns_tpr_te, _ = roc_curve(test_labels, ns_probs_te)
    clf_fpr_te, clf_tpr_te, _ = roc_curve(test_labels, clf_probs_te)


    clf_precision_te, clf_recall_te, _ = precision_recall_curve(test_labels, clf_probs_te)
    clf_f1_te, clf_auc_te = f1_score(test_labels, pred_labels_test), auc(clf_recall_te, clf_precision_te)
    # summarize scores
    print('f1=%.3f auc=%.3f' % (clf_f1_te, clf_auc_te))
    
    no_skill = len(test_labels[test_labels==1]) / len(test_labels)

    # plot 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.title.set_text('ROC AUC')
    ax2.title.set_text('Precision-Recall')

    ax1.plot(ns_fpr_te, ns_tpr_te, linestyle='--', label='No Skill')
    ax1.plot(clf_fpr_te, clf_tpr_te, marker='.', label='BDT')

    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    ax1.legend()

    ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax2.plot(clf_recall_te, clf_precision_te, marker='.', label='BDT')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f'../figs/{prefix}test_scores{suffix}.pdf')

    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Training Data ***************')
    score_tr = estimator.score(train_features, train_labels)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(train_labels, pred_labels_train))

    plot_confusion_matrix(estimator, train_features, pred_labels_train, labels=[1,0], cmap='GnBu')  
    plt.savefig(f'../figs/{prefix}train_conf_mtx{suffix}.pdf')


    ### ROC-AUC + precision-recall
    ns_probs_tr = [0 for _ in range(len(train_labels))]
    clf_probs_tr = estimator.predict_proba(train_features)
    clf_probs_tr = clf_probs_tr[:, 1]
    
    # calculate scores
    ns_rauc_tr = roc_auc_score(train_labels, ns_probs_tr)
    clf_rauc_tr = roc_auc_score(train_labels, clf_probs_tr)
    # summarize scores
    print('ROC AUC=%.3f' % (clf_rauc_tr))
    # calculate roc curves
    ns_fpr_tr, ns_tpr_tr, _ = roc_curve(test_labels, ns_probs_te)
    clf_fpr_tr, clf_tpr_tr, _ = roc_curve(test_labels, clf_probs_te)

    clf_precision_tr, clf_recall_tr, _ = precision_recall_curve(train_labels, clf_probs_tr)
    clf_f1_tr, clf_auc_tr = f1_score(train_labels, pred_labels_train), auc(clf_recall_tr, clf_precision_tr)
    # summarize scores
    print('f1=%.3f auc=%.3f' % (clf_f1_tr, clf_auc_tr))

    no_skill = len(train_labels[train_labels==1]) / len(train_labels)

    # plot 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.title.set_text('ROC AUC')
    ax2.title.set_text('Precision-Recall')

    ax1.plot(ns_fpr_tr, ns_tpr_tr, linestyle='--', label='No Skill')
    ax1.plot(clf_fpr_tr, clf_tpr_tr, marker='.', label='BDT')
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    ax1.legend()
    
    ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax2.plot(clf_recall_tr, clf_precision_tr, marker='.', label='BDT')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f'../figs/{prefix}train_scores{suffix}.pdf')
    print('--------------------------------------------------------')
    print("")


# -------------------------------------------------------------------------------------
# Body
# -------------------------------------------------------------------------------------


df = pd.read_excel('../event_class.xlsx', sheet_name=0, header=0)
df = model_preprocessing(df)

print('"event" class has a total of {} rows'.format(df.query('classifier == 1').shape[0]))
print('"non-event" class has a total of {} rows'.format(df.query('classifier == 0').shape[0]))

event_df = df.query('classifier == 1')
nonevent_df = df.query('classifier == 0')


correlation_blocks(df, 'correlation_blocks.pdf', 'XGB_')
correlation_blocks(event_df, 'event_correlation_blocks.pdf', 'XGB_')
correlation_blocks(nonevent_df, 'nonevent_correlation_blocks.pdf', 'XGB_')


# delete features with correlation >= 0.9
corr = df.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]


plot_column_hist(df, event_df, nonevent_df, 'XGB_')



X = df.drop(columns='classifier')
y = df['classifier']

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


model = XGBClassifier()
clf = model.fit(X_train, y_train)

print(model)

y_pred_te = model.predict(X_test)
y_pred_tr = model.predict(X_train)

model_metrics(clf, model, X_train, y_train, X_test, y_test, y_pred_te, y_pred_tr, prefix='XGB_')



#Find the optimal value with optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
 
#Fits tuned hyperparameters
opt_xgb = XGBClassifier(max_depth=study.best_params['max_depth'], n_estimators=study.best_params['n_estimators'], tree_method='exact', objective='binary:logistic')
# opt_xgb = XGBClassifier(eta=study.best_params['eta'], gamma=study.best_params['gamma'], min_child_weight=study.best_params['min_child_weight'],
#                             max_delta_step=study.best_params['max_delta_step'], subsample=study.best_params['subsample'],
#                             tree_method='exact', objective='binary:logistic')
 

opt_clf = opt_xgb.fit(X_train ,y_train)

y_pred_te_opt = opt_xgb.predict(X_test)
y_pred_tr_opt = opt_xgb.predict(X_train)

model_metrics(opt_clf, opt_xgb, X_train, y_train, X_test, y_test, y_pred_te_opt, y_pred_tr_opt, prefix='XGB_', suffix='_opt')