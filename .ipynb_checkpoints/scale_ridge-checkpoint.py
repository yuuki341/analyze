import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

#データの読み込み
train_x = pd.read_csv('train_x.csv')
train_y = pd.read_csv('train_y.csv')

#すべてがnanを削除
train_x=train_x.dropna(how='all',axis=1)
 
#すべての要素が同じものを削除
delete=['掲載期間　終了日','動画タイトル','動画コメント','（派遣）応募後の流れ','対象者設定　年齢上限','動画ファイル名',
 '産休育休取得事例あり','勤務地固定','週1日からOK','ミドル（40〜）活躍中','ルーティンワークがメイン','対象者設定　年齢下限',
 '土日祝のみ勤務','CAD関連のスキルを活かす','固定残業制','公開区分','20代活躍中','検索対象エリア','就業形態区分',
 '30代活躍中','雇用形態','Dip JobsリスティングS','資格取得支援制度あり','社会保険制度あり','残業月10時間未満','履歴書不要',
 '研修制度あり','DTP関連のスキルを活かす','新卒・第二新卒歓迎','対象者設定　性別','WEB登録OK']
train_x=train_x.drop(delete,axis=1)

#要素としていらないものを削除
dele=['掲載期間　開始日','期間･時間　備考','（派遣先）勤務先写真ファイル名','勤務地　最寄駅2（分）','期間・時間　勤務開始日',
      '（派遣先）概要　事業内容',
      '勤務地　最寄駅2（駅名）','（紹介予定）雇用形態備考','勤務地　最寄駅2（沿線名）','休日休暇　備考','期間・時間　勤務時間',
         '勤務地　備考','拠点番号','（紹介予定）入社時期','お仕事名','（派遣先）配属先部署','仕事内容','（紹介予定）年収・給与例',
         '勤務地　最寄駅1（沿線名）','応募資格','（紹介予定）休日休暇','派遣会社のうれしい特典','（派遣先）職場の雰囲気',
         '（紹介予定）待遇・福利厚生','給与/交通費　備考','お仕事のポイント（仕事PR）','（派遣先）概要　勤務先名（漢字）',
      '職種コード','お仕事No.','会社概要　業界コード','勤務地　市区町村コード','勤務地　最寄駅1（駅名）','職場の様子','（紹介予定）入社後の雇用形態','給与/交通費　給与支払区分','フラグオプション選択',
         '勤務地　最寄駅2（駅からの交通手段）','勤務地　都道府県コード','仕事の仕方','給与/交通費　交通費']
train_x=train_x.drop(dele,axis=1)

#nanを0にする
train_x=train_x.fillna(0)

#特徴量スケーリング
scaler = MinMaxScaler()
scaler.fit(train_x)
X_train_scaled = scaler.transform(train_x)

#整形
X_train=X_train.values
y_train=train_y.values
y_train=y_train[:,1]

#データの予測
ridge = Ridge(alpha=100,normalize=False, solver="svd") 
ridge.fit(X_train, y_train)

# モデルを保存する
filename = 'ridge_model.sav'
pickle.dump(model, open(filename, 'wb'))
