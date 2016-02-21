#Damir Jajetic, 2016, MIT licence

def predict (LD, output_dir, basename):
	import copy
	import os
	import numpy as np
	import libscores
	import data_converter
	from sklearn import preprocessing, ensemble
	from sklearn.utils import shuffle

	
	LD.data['X_train'], LD.data['Y_train'] = shuffle(LD.data['X_train'], LD.data['Y_train'] , random_state=1)
	
	Y_train = LD.data['Y_train']
	X_train = LD.data['X_train']
	
	Xta = np.copy(X_train)

	X_valid = LD.data['X_valid']
	X_test = LD.data['X_test']
	
	
	Xtv = np.copy(X_valid)
	Xts = np.copy(X_test)
	

	import xgboost as xgb
	if LD.info['name']== 'alexis':

		model = ensemble.RandomForestClassifier(max_depth=140, n_estimators=1800, n_jobs=-1, random_state=0, verbose=0, warm_start=True)
		model2 = ensemble.RandomForestClassifier(max_depth=140, n_estimators=1800, n_jobs=-1, random_state=1, verbose=0, warm_start=True)
		model.fit(X_train, Y_train)	
		model2.fit(X_train, Y_train)
		
		preds_valid0 = model.predict_proba(X_valid)
		preds_test0 = model.predict_proba(X_test)
		
		preds_valid2 = model2.predict_proba(X_valid)
		preds_test2 = model2.predict_proba(X_test)
		
		preds_valid0 = np.array(preds_valid0)
		preds_valid2 = np.array(preds_valid2)
		
		preds_test0 = np.array(preds_test0)
		preds_test2 = np.array(preds_test2)
		
		
		preds_valid = (preds_valid0 + preds_valid2)/2
		preds_test = (preds_test0 + preds_test2)/2
		
		
		preds_valid = preds_valid[:, :, 1]
		preds_valid = preds_valid.T
		
		
		preds_test = preds_test[:, :, 1]
		preds_test = preds_test.T
		
		
	if LD.info['name']== 'dionis': 
		Lest = 600 #600 will consume cca 250 GB of RAM, use 50 for similar result
		#Lest = 50
		
		model = ensemble.RandomForestClassifier( n_jobs=-1, n_estimators=Lest, random_state=0)
		model.fit(X_train, Y_train)	
		preds_valid0 = model.predict_proba(X_valid)
		preds_test0 = model.predict_proba(X_test)

		model = ensemble.RandomForestClassifier( n_jobs=-1, n_estimators=Lest, random_state=1)
		model.fit(X_train, Y_train)	
		preds_valid1 = model.predict_proba(X_valid)
		preds_test1 = model.predict_proba(X_test)
		
		model = ensemble.RandomForestClassifier( n_jobs=-1, n_estimators=Lest, random_state=2)
		model.fit(X_train, Y_train)	
		preds_valid2 = model.predict_proba(X_valid)
		preds_test2 = model.predict_proba(X_test)
		
		model = ensemble.RandomForestClassifier( n_jobs=-1, n_estimators=Lest, random_state=3)
		model.fit(X_train, Y_train)	
		preds_valid3 = model.predict_proba(X_valid)
		preds_test3 = model.predict_proba(X_test)
		
		model = ensemble.RandomForestClassifier( n_jobs=-1, n_estimators=Lest, random_state=4)		
		model.fit(X_train, Y_train)	
		preds_valid4 = model.predict_proba(X_valid)
		preds_test4 = model.predict_proba(X_test)
		
		preds_valid = (preds_valid0 + preds_valid1 + preds_valid2 + preds_valid3 + preds_valid4) # /5 should be included (bug)
		preds_test = (preds_test0 + preds_test1 + preds_test2 + preds_test3 + preds_test4) # /5 should be included (bug)
	
	
	if LD.info['name']== 'grigoris':
		model = ensemble.RandomForestClassifier(criterion='entropy', max_features=0.05, max_depth=5, n_estimators=120, n_jobs=-1, random_state=0, verbose=0)
		model2 = linear_model.LogisticRegression(penalty='l1', random_state=1, n_jobs=-1, C=0.008)
		model3 = ensemble.RandomForestClassifier(criterion='entropy', max_features=0.05, max_depth=5, n_estimators=120, n_jobs=-1, random_state=1, verbose=0)
		model4 = ensemble.RandomForestClassifier(criterion='entropy', max_features=0.05, max_depth=5, n_estimators=120, n_jobs=-1, random_state=2, verbose=0)
		
		preds_valid = np.zeros((X_valid.shape[0], Y_train.shape[1]))
		preds_test = np.zeros((X_test.shape[0], Y_train.shape[1]))
		for pyt in range(Y_train.shape[1]):
			print pyt
			ytp = Y_train[:, pyt]
			model.fit(X_train, ytp)				
			model2.fit(X_train, ytp)
			model3.fit(X_train, ytp)
			model4.fit(X_train, ytp)
			
			preds1v= model.predict_proba (X_valid)[:, 1]
			preds2v= model2.predict_proba (X_valid)[:, 1]
			preds3v= model3.predict_proba (X_valid)[:, 1]
			preds4v= model4.predict_proba (X_valid)[:, 1]
			predsv = (preds1v + preds2v + preds3v + preds4v)/4
			preds_valid[:, pyt] = predsv
			
			preds1t= model.predict_proba (X_test)[:, 1]
			preds2t= model2.predict_proba (X_test)[:, 1]
			preds3t= model3.predict_proba (X_test)[:, 1]
			preds4t= model4.predict_proba (X_test)[:, 1]
			predst = (preds1t + preds2t + preds3t + preds4t)/4
			preds_test[:, pyt] = predst

			
	if LD.info['name']== 'jannis':	
		Xd = X_train[Y_train==0]
		yd = Y_train[Y_train==0]
	
		for a in range(18):
			X_train = np.vstack([X_train, Xd])
			Y_train = np.hstack([Y_train, yd])
		
	
		Xd = X_train[Y_train==2]
		yd = Y_train[Y_train==2]
	
	
		X_train = np.vstack([X_train, Xd])
		Y_train = np.hstack([Y_train, yd])
		
		Y_train_raw = np.array(data_converter.convert_to_bin(Y_train, len(np.unique(Y_train)), False))
		
		
		preds_valid = np.zeros((X_valid.shape[0], Y_train_raw.shape[1]))
		preds_test = np.zeros((X_test.shape[0], Y_train_raw.shape[1]))
		for pyt in range(Y_train_raw.shape[1]):
			if pyt == 0:
				Lbs = 0.2
			else:
				Lbs = 0.5

			model = xgb.XGBClassifier(max_depth=30, learning_rate=0.05, n_estimators=100, silent=True, 
				objective='binary:logistic', nthread=-1, gamma=0, 
				min_child_weight=80, max_delta_step=1, subsample=1, 
				colsample_bytree=1, base_score=Lbs, seed=0, missing=None)

					
			ytp = Y_train_raw[:, pyt]
			model.fit(X_train, ytp)
			
			
			preds1v= model.predict_proba (X_valid)[:, 1]
			preds_valid[:, pyt] = preds1v 
			
			preds1t= model.predict_proba (X_test)[:, 1]
			preds_test[:, pyt] = preds1t
	
		
	if LD.info['name']== 'wallis':
		model = naive_bayes.MultinomialNB(alpha=0.02)
		
		model2 = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=1200, silent=True, 
				objective='multi:softprob', nthread=-1, gamma=0, 
				min_child_weight=1, max_delta_step=0, subsample=1, 
				colsample_bytree=1, base_score=0.5, seed=0, missing=None)
				
	
		model.fit(X_train, Y_train)
		preds_valid1 = model.predict_proba(X_valid)		
		preds_test1 = model.predict_proba(X_test)
		
		model2.fit(X_train, Y_train)
		preds_valid2 = model2.predict_proba(X_valid)
		preds_test2 = model2.predict_proba(X_test)
				
		preds_valid = (preds_valid1 +preds_valid2)/2
		preds_test = (preds_test1 +preds_test2)/2				
	
	import data_io
	if  LD.info['target_num']  == 1:
		preds_valid = preds_valid[:,1]
		preds_test = preds_test[:,1]
								
	preds_valid = np.clip(preds_valid,0,1)
	preds_test = np.clip(preds_test,0,1)
	
	data_io.write(os.path.join(output_dir, basename + '_valid_000.predict'), preds_valid)
	data_io.write(os.path.join(output_dir,basename + '_test_000.predict'), preds_test)

