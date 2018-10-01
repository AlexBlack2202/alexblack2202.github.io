---
type: post
title: Xây dựng chương trình gợi ý phim dựa vào tập dữ liệu movie len
date: 2018-10-01 00:19:00 +0300
description: Ở bài viết này, tôi sẽ tập trung vào vào sử dụng implement Alternating Least Saqures của Collaborative Filtering trong thư viện Spark MLlib trên tập dữ liệu movieLens. # Add post description (optional)
imgtitle: ../../post_image/AlexNet-1.png # Add image post (optional)
tags: [Machine learning, Deeplearning, Spark] # add tag
---

### Lời mở đầu

MovieLens là một tập dữ liệu được sử dụng rộng rãi cách đây nhiều năm. Hôm nay, tôi sẽ sử dụng tập dữ liệu này và mô hình ALS để xây dựng chương trình dự đoán phim cho người dùng. 

### Chuẩn bị dữ liệu

Các bạn có thể download tập dữ liệu MovieLens ở link https://grouplens.org/datasets/movielens/. Các bạn có thể download trực tiếp 2 file nén ở link http://files.grouplens.org/datasets/movielens/ml-latest-small.zip và link  http://files.grouplens.org/datasets/movielens/ml-latest.zip.

Ở trên bao gồm 2 tập dữ liệu. Trong python, chúng ta tạo thư mục datasets và download rồi bỏ chúng vào trong thư mục đấy.

```python
complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

import os

datasets_path = 'datasets'
if not os.path.exists(datasets_path):
    os.makedirs(datasets_path))

complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')

import urllib
import zipfile

if not os.path.exists(small_dataset_url):
	small_f = urllib.urlretrieve (small_dataset_url, small_dataset_path)#Download
	with zipfile.ZipFile(small_dataset_path, "r") as z:#Giải nén
		z.extractall(datasets_path)
if not os.path.exists(small_dataset_url):
	complete_f = urllib.urlretrieve (complete_dataset_url, complete_dataset_path)#Download
	with zipfile.ZipFile(complete_dataset_path, "r") as z:#Giải nén
		z.extractall(datasets_path)

```

Trong thư mục giải nén, chúng ta sẽ có các file ratings.csv, movies.csv, tags.csv, links.csv, README.txt. 

### Loading và parsing dữ liệu.

Mỗi dòng trong tập ratings.csv có định dạng `"userId,movieId,rating,timestamp"`.

Mỗi dòng trong tập movies.csv có định dạng `"movieId,title,genres"`.

Mỗi dòng trong tập tags.csv có định dạng `"userId,movieId,tag,timestamp"`.

Mỗi dòng trong tập links.csv có định dạng `"movieId,imdbId,tmdbId"`.

Tóm lại, các trường dữ liệu trong các file csv đều ngăn cách nhau bởi dấu phẩy (,). Trong python, ta có thể dùng hàm split để cắt chúng ra. Sau đó sẽ load toàn bộ dữ liệu lên RDDs.

Lưu ý nhỏ:

* Ở tập dữ liệu ratings, chúng ta chỉ giữ lại các trường `(UserID, MovieID, Rating)` bỏ đi trường timestamp vì không cần thiết.
* Ở tập dữ liệu movies  chúng ta giữ lại trường `(MovieID, Title)` và bỏ đi trường genres vì lý do tương tự.

```python
small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
print(small_ratings_data.take(3)) #Hiện thị top 3 ratting đầu tiên

small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
    
small_movies_data.take(3) #Hiện thị top 3 movie đầu tiên
```

Phần tiếp theo, chúng ta sẽ tìm hiểu lọc cộng tác (Collaborative Filtering) và cách sử dụng Spark MLlib để xây dựng mô hình dự báo. 

### Collaborative Filtering

Ở đây, tôi sẽ không đề cập đến lọc cộng tác là gì, các bạn có nhu cầu tìm hiểu có thể xem ở bài post khác hoặc tham khảo trên wiki. Chúng ta sẽ tập trung vào tìm hiểu cách sử dụng ALS trong thư viện MLlib của Spark. Các tham số của thuật toán này bao gồm:

* numBlocks: số lượng block được sử dụng trong tính toán song song (-1 với ý nghĩa là auto configure).

* rank: số lượng nhân tố ẩn (latent factor) trong mô hình.

* iterations: số lần lặp.

* lambda: tham số của chuẩn hoá(regularization ) trong ALS.

* implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.

* alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

### Chọn các tham số cho ALS

Để chọn được các tham số tốt nhất cho mô hình ALS, chúng ta sẽ sử dụng tập small để grid search. Đầu tiên, chúng ta chia tập dữ liệu thành 3 phần là tập train, tập vali và  tập test. Sau đó tiến hành huấn luyện trên tập train và predict trên tập valid để tìm được tham số tốt nhất. Cuối cùng đánh giá kết quả đạt được trên tập test.

```python
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

from pyspark.mllib.recommendation import ALS
import math

seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print('The best model was trained with rank %s' % best_rank)
```

Kết quả sau khi thực hiện đoạn code trên là:

```python
For rank 4 the RMSE is 0.963681878574
For rank 8 the RMSE is 0.96250475933
For rank 12 the RMSE is 0.971647563632
The best model was trained with rank 8
```

Tiến hành thực hiện test

```python
model_test = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model_test.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print('For testing data the RMSE is %s' % (error))
```

```python
For testing data the RMSE is 0.972342381898
```

Xem kỹ hơn một chút về dữ liệu mà spark trả về cho chúng ta. Với predictions và rates_and_preds, ta có:

```python
print(predictions.take(3))
```

```python
[((32, 4018), 3.280114696166238),
 ((375, 4018), 2.7365714977314086),
 ((674, 4018), 2.510684514310653)]
```

Tập dữ liệu trả về bao gồm cặp `(UserID, MovieID)` và `Rating` (tương ứng với colum 0, column 1 và column 2 ở trên),được hiểu ở đây là với người dùng UserID và phim MovieID thì mô hình sẽ dự đoán người dùng sẽ rating kết quả Rating.

Sau đó chúng ta sẽ nối(join) chúng với tập valid tương ứng theo cặp `(UserID, MovieID)`, kết quả đạt được là:
```python
rates_and_preds.take(3)
```

```python
[((558, 788), (3.0, 3.0419325487471403)),
 ((176, 3550), (4.5, 3.3214065001580986)),
 ((302, 3908), (1.0, 2.4728711204440765))]
 ```
 
Việc còn lại là chúng ta sẽ tính trung bình độ lỗi bằng hàm `mean()` và `sqlt()`


### Xây dựng mô hình với tập dữ liệu large

Tiếp theo, chúng ta sẽ sử dụng tập dự liệu bự hơn để xây dựng mô hình. Cách thực hiện y chang như tập dữ liệu nhỏ đã được trình bày ở trên, nên tôi sẽ bỏ qua một số giải thích không cần thiết để tránh lặp lại.

```python
# Load the complete dataset file
complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Parse
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print("There are %s recommendations in the complete dataset" % (complete_ratings_data.count()))
```

```python
There are 21063128 recommendations in the complete dataset
```

Tiến hành train và test.

```python
training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0)

complete_model = ALS.train(training_RDD, best_rank, seed=seed,iterations=iterations, lambda_=regularization_parameter)

test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print('For testing data the RMSE is %s' % (error))
```

```python
For testing data the RMSE is 0.82183583368
```












