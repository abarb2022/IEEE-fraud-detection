# IEEE-fraud-detection
## კონკურსის მოკლე მიმოხილვა
IEEE-fraud-detection კონკურსში მიზანია ML მოდელის შექმნა, რომელიც ტრანზაქციებშ თაღლითობას გამოვლენს. მონაცემები წამოსულია რეალური ტრანზაქციებიდან და მოიცავს მრავალ ფაქტორს (მოწყობილობის ტიპი, პროდუქტის მახასიათებლები და ა.შ.) გვაქვს ორი დატასეტი და მონაცემები დაუბალანსებელია, თაღლითობა (ანუ 1ები) ბევრად ჭარბობენ 0ებს.
## ჩემი მიდგომა
მე ვეცადე Nan-ების ეფექტურად ჩანაცვლებისა და categorical ცვლადების numerical-ში გადაყვანის სხვადასხვა გზები დამეტესტა. ყველა მოდელში ვშლი მაღალი კორელაციის fetature-ებს. დავტესტე 4 არქიტექტურა: logistic regression
, Random Forest, XGBoost, ADABoost. 
## ფაილები
model_experiment_logistic_regression.ipynb - logistic regression არქიტექტურის მოდელის ტრენინგი
model_experiment_adaboost.ipynb ...
model_experiment_xgboogst.ipynb  ...
model_experiment_random_forest.ipynb ...
model_inference - ამ ფაილში ხდება test set ზე პროგნოზი, საუკეთესო მოდელის გამოყენებით ვაგენერირებ submissions. 

## data split
დატა train/validation/test-ად დავყვავი card1-ის დაგრუპვით. ასე ჩვენი სეტები უფრო რეალურია და ამასთან data leakage-ს გვარიდებს თავიდან, ერთი კარტის მფლობელი სხვადასხვა სეტში აღარ შეგვხვდება. 

## logistic regression

### Feature Engineering
თავიდან Nan ცვალდებს ვამუშავებ მარტივად, ვავსებ მოდით. კატეგორიულების გადასაყვანად გამოვიყენე TargetEncoder
## Feature Selection
ვფილტრავ მაღალკოლერიებულ და დაბალი ვარიაციის მქონე featureb-ს.

### Training
ვიყენებ undersampling-ს, რადაგან დატა არის დაუბალანსებელი და ასევე ცოტა ოპტიმიზაციისთვის დავამატე StandardScaler. gridsearch გავუშვი რეგულარიზაციის სიმძლავრეზე და 0.01 აირჩა. შდეგები კი ასეთი იყო:
val_roc_auc 0.8502018030597784
train_roc_auc 0.8541808770151914
test_roc_auc 0.7968875668130505

გამოსასწორებლად  Nan ცვალდების დასამუშავებლად შევქმენი კლასი - SmartMissingHandler, რომელიც დროპავს იმ სვეტებს, რომლებში 80 პროცენტზე მეტი Nan ხვდება. დანარჩენ ნანებს კი ვყოფ ორ ნაწილად. ზოგიერთ შემთხვევაში კონკრეტული Features Nan-ობა კოლერილებულია fraud-ობასთან, ამის გამო ასეთ featurebს ვავსებ -999/'MISSING' -ით. ხოლო ნაკლებად კოლერილებულებს ვავსებ მოდით. დანარჩენი კი ყველაფერი იგივე დავტოვე, ამან 0.1-ით გააუმჯობესა შედეგი ტესტზე (0.8006269733193636)


## Random Forest
აქაც გამოვიყენე SmartMissingHandler და TargetEncoder, გავფილტრე კორელაციით და დაბალი ვარიაციით და გრიდსერჩით ჯერ შევარჩიე sampling_strategy, საუკეთესო მომცა 0.2-მა :
val_roc_auc  0.849192424768182
train_roc_auc 0.8470842537706055
test_roc_auc 0.8005435687404019
შემდეგ ეგ ჰიპერპარამეტრი დავუსეტე და ისევ გრიდსერჩით გადავარჩიე max_depth(საუკეთესო -15):
val_roc_auc 0.8859214997088913
train_roc_auc 0.9438897315999673
test_roc_auc 0.8217496823153074
gridsearch-მა მოდელი overfit-ში წაიყვანა, train-მა ვალიდაციასთან შედარებით ზედმეტად დაისწავლა


## XGBoost:
წინა ორ მოდელში მხოლოდ train_transaction.csv-ს ვიყენებდი. აქ left join გავუკეთე train_identity.csv-თან. აქ თავიდან გამოვიყენე ჩვეულებრივი Nan_handler, რომელიც ბევრ Nan-იანებს დროპავს და დანარჩენებს მოდით ანაცვლებს, გამოვიყენე CustomPreprocessor, აქ ვჰენდლავ numerical მნიშვნელობებს. 3-ზე ნაკლები უნიკალურ მნიშვნელობიანი სვეტებისთვის ვიყენებ one-hot encodingს. ხოლო მეტიანებზე გამოვიყენე Ordinal Encoding. დანარჩენი იგივე დავტოვე. XGBoost ასეთ გაშვებაზე წავიდა overfit-ში (grid_search- ით ამოვარჩიე უკეთესი max_depth):
train_roc_auc 0.9819756483969074
test_roc_auc 0.8507624249701462
Nan_handlerი-ს SmartMissingHandler-ით ჩანაცვლებით ტესტზე შედეგი გაზიარდა 0.01-ით.
მესამე run-ზე მთლიანად ordinalEncoding-ის დახმარებით ვეცადე შემეცირებინა overfit: 
train_roc_auc 0.9141405781600476
test_roc_auc 0.8471468851290882

## AdaBoost:
ყველაზე ნაკლები ოვერფიტი დაიდო აქ, (Ordinal Encoding, SmartMissingHandler) + ვარიაცია&კოლერაციის ფილტრი.:
Train ROC-AUC: 0.8587
Test ROC-AUC: 0.8219
საბოლოოდ ავირჩიე ადაბუსტი, რადგან ტრეინშიც ყველაზე ნაკლები ოვერფიტი დადო და submission-ის გაკეთებისასაც ნორმალური ქულა ჰქონდა.
public Score: 0.882633
Private score: 0.858907

## mlflow tracking - ბმულები
logistic regression - https://dagshub.com/abarb22/IEEE-fraud-detection.mlflow/#/experiments/2?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
Random Forest - https://dagshub.com/abarb22/IEEE-fraud-detection.mlflow/#/experiments/1?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
XGBoost - https://dagshub.com/abarb22/IEEE-fraud-detection.mlflow/#/experiments/3?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
ADABoost - https://dagshub.com/abarb22/IEEE-fraud-detection.mlflow/#/experiments/4?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
