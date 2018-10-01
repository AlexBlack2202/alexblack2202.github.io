---
type: post
title: Tìm hiểu về mạng neural network AlexNet
date: 2018-06-15 00:19:00 +0300
description: Bài viết này được mình lược dịch từ nguồn của tác giả SUNITA NAYAK. Nội dung chủ yếu là tìm hiểu sâu về kiến trúc mạng AlexNet. # Add post description (optional)
imgtitle: ../../post_image/AlexNet-1.png # Add image post (optional)
tags: [Machine learning, Deeplearning, AlexNet] # add tag
---

### Lời mở đầu

Tỷ phú Peter Thiel đã từng đưa ra câu hỏi tréo nghoe như thế này: "What important truth do very few people agree with you on?"

Nếu bạn đen câu này hỏi giáo sư Geoffrey Hinton vào năm 2010, ông ấy sẽ trả lời rằng mạng Convolutional Neural Networks (CNN) sẽ có bước đột phá lớn và giúp chúng ta giải quyết hoàn toàn bài toán phân loại ảnh. Tại thời điểm năm 2010, các nhà nghiên cứu trong lĩnh vực phân loại ảnh đều không nghĩ như giáo sư Geoffrey Hinton. Và Deep Learning tại thời điểm đó chưa thật sự giải quyết bài toán này. 

Năm 2010 cũng là năm ra đời của cuộc thi ImageNet Large Scale Visual Recognition Challenge. Tập dữ liệu ảnh trong cuộc thi bao gồm khoảng 1.2 triệu ảnh thuộc 1 ngàn lớp khác nhau, người thắng cuộc là người tạo ra mô hình làm cho độ lỗi trên tập dữ liệu trên là nhỏ nhất.

Hai năm sau, trong bài báo cáo "ImageNet Classification with Deep Convolutional Neural Networks" của nhóm tác giả Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, Geoffrey và các cộng sự của mình đã chứng minh điều ông ấy nói hai năm trước là hoàn toàn chính xác.

Trong bài báo này, nhóm các giả đã huấn luyện mạng CNN và và đạt độ lỗi top-5 error rate là 15.3% (hạng nhất), cách biệt khá xa so với kết quả của nhóm đứng thứ hai( độ lỗi 26.2%). Trong các năm tiếp theo, rất nhiều nhóm đã nghiên cứu, cải tiến kiến trúc của mô hình CNN để đạt được kết quả tốt hơn, thậm chí hơn luôn khả năng nhận biết của con người.

Kiến trúc mạng CNN được sử dụng vào năm 2012 được cộng đồng nghiên cứu gọi với tên gọi thân thương là AlexNet do tác giả chính của nhóm nghiên cứu là Alex Krizhevsky. Ở trong bài viết này, chúng ta sẽ đi sâu vào tìm hiểu kiến trúc AlexNet và đóng góp chính của nó trong CNN.

### Đầu vào

Như đã đề cập ở phần trên, mạng AlexNet đã thắng hạng nhất trong cuộc thi ILSVRC năm 2012. Mô hình giải quyết bài toán phân lớp một bức ảnh vào 1 lớp trong 1000 lớp khác nhau (vd gà, chó, mèo ... ). Đầu ra của mô hình là một vector có 1000 phần tử. Phần tử thứ i của vector đại diện cho xác suất bức ảnh thuộc về lớp thứ i. Do đó, tổng của các phần tử trong vector là 1.

Đầu vào của mạng AlexNet là một bức ảnh RGB có kích thước 256x256 pixel. Toàn bộ các bức ảnh của tập train và tập test đều có cùng kích thước là 256x256. Nếu một bức ảnh nào đó không có kích thước 256x256, bức ảnh đó sẽ được chuyển về kích thước đúng 256x256. Bằng cách, những bức hình có kích thước nhỏ hơn 256 thì sẽ được phóng bự lên đến kích thước 256. Những bức hình nào có kích thước lớn hơn 256 thì sẽ được cắt loại phần thừa để nhận được bức hình có kích thước 256x256. Hình ảnh ở dưới là một ví dụ về việc điều chỉnh bức ảnh về kích thước 256x256. 

![Hình ảnh](/post_image/AlexNet-Resize-Crop-Input.jpg)

Nếu ảnh đầu vào là ảnh xám (grayscale), bức ảnh trên sẽ được chuyển đổi thành định dạng RGB bằng cách tạo ra 3 layer kênh màu giống nhau từ ảnh xám.
Sau khi chuẩn hoá hết tất cả các ảnh về dạng 256x256x3, nhóm tác giả chỉ sử dụng một phần có kích thước 227x227x3 của một bức ảnh làm đầu vào cho mạng neural network. Trong bài báo nhóm tác giả ghi là 224x224, nhưng đây là một lỗi nhỏ của nhóm tác giả, và kích thước thực tế đầu vào của bức ảnh là 227x227.

### Kiến trúc AlexNet
Kiến trúc AlexNet lớn hơn nhiều so với các kiến trúc CNNs được sử dụng trong thị giác máy tính trước kia, vd kiến trúc LeNet của Yann LeCun năm 1998. Nó có 60 triệu tham số và 650000 neural và tốn khoảng từ năm đến sáu ngày huấn luyện trên hai GPU GTX 580 3GB. Ngày nay, do sự tiến bộ vượt bật của GPU, chúng ta có nhiều kiến trúc CNN có cấu trúc phức tạp hơn, nhưng vẫn hoạt động rất hiệu quả trên những tập dữ liệu phức tạp. Nhưng tại thời điểm hiện năm 2012 thì việc huấn luyện mô hình với lượng tham số và neural lớn như vậy là một vấn đề cực kỳ khó khăn. Nhìn kỹ vào hình bên dưới để hiểu rõ hơn về kiến trúc AlexNet.
![Kiến trúc AlexNet](/post_image/AlexNet-1.png)

AlexNet bao gồm 5 convolution Layer và 3 Fully connected Layers.

Những convolution layer ( hay còn gọi với tên khác là các filter) rút trích các thông tin hữu ích trong các bức ảnh. Trong một convolution layter bất kỳ thường bao gồm nhiều kernel có cùng kích thước. Ví dụ như convolution layer đầu tiên của AlexNet chứa 96 kernel có kích thước 11x11x3. Thông thường thì width và height của một kernel thường bằng nhau, và độ sâu (depth) thường bằng số lượng kênh màu.

Convolutional 1 và convolution 2 kết nối với nhau qua một Overlapping Max Pooling ở giữa. Tưng tự như vậy giữa convolution 2 và convolution 3. Convolutional 3, convolution 4, convolution 5 kết nối trực tiếp với nhau, không thông qua trung gian. Convolutional 5 kết nối fully connected layter 1 thông qua một Overlapping Max pooling, Tiêp theo mà một fully connected layter nữa. Và cuối cùng là một bộ phân lớp softmax   với 1000 lớp nhãn.

ReLU nonlinerity được sử dụng sau tất các các convolution và fully connected layer.Trước đây, ReLU nonlinerity của lớp convolution 1 và 2 thường theo sau bởi một bước chuẩn hoá cục bộ (local normalization) rồi mới thực hiện pooling. Tuy nhiên, các nghiên cứu sau đó nhận thấy rằng việc sử dụng normalization không thật sự hữu ích. Do vậy chúng ta sẽ không đi chi tiết về vấn đề đó.

### Overlapping Max Pooling

Max Pooling layter thường được sử dụng để giảm chiều rộng và chiều dài của một tensor, giữ nguyên chiều sâu. Overlapping Max Pool layter cũng tương tự như Max Pool layter, ngoại trừ việc là một window của bước này sẽ có một phần chồng lên window của bước tiếp theo. Tác giả sử dụng pooling có kích thước 3x3 và bước nhảy là 2 giữa các pooling. Nghĩa là giữa pooling này và pooling khác sẽ overlapping với nhau 1 pixel. Các thí nghiệm thực tế đã chứng minh rằng việc sử dụng overlapping giữa các pooling giúp giảm độ lỗi top-1 error 0.4% và top-5 error là 0.3% khi so với việc sử dụng pooling có kích thước 2x2 và bước nhảy 2 (vector output của cả hai đều có số chiều bằng nhau). 

### ReLu Nonlinearity

Một cải tiến quan trọng khác của AlexNet là việc sử dụng hàm phi tuyến ReLU. Trước đây, các nhóm nghiên cứu khác thường sử dụng hàm kích hoạt là hàm Tanh hoặc hàm Sigmoid để huấn luyên mô hình neural network. AlexNet chỉ ra rằng, khi sử dụng ReLU, mô hình deep CNN sẽ huấn luyện nhanh hơn so với viêc sử dụng tanh hoặc sigmoid. Hình bên dưới được rút ra từ bài báo chỉ ra rằng với việc sử dụng ReLU (đường nét liền trong hình), AlexNet đạt độ lỗi 25% trên tập huấn luyện nhan hơn gấp 6 lần so với mô hình tương tự nhưng sử dụng Tanh (đường nét đứt trong hình). Thí nghiệm trên sử dụng tập dữ liệu CIFAR-10 để huấn luyện.

![Tốc độ hội tụ của mạng AlexNet](/post_image/ReluNonlinearity-768x635.png)

Để hiểu rõ hơn lý do vì sao ReLU lại nhanh hơn so với các hàm khác, chúng ta hãy đối sánh hình dạng giá trị output của các hàm trên.

Công thức của ReLU là: f(X) = max(0,x)

![Hàm kích hoạt của ReLU và tanh](/post_image/Tanh-300x238.png) 

Nhìn kỹ vào hình trên, ta có nhận xét rằng: hàm tanh đạt giá trị bão hoà khi giá trị z >2.5 và z < -2.5 (số 2.5 là số cảm tính của mình). Và tại vùng \|z\|>2.5, thì độ dốc của hàm hầu như gần như bằng 0, \|z\| càng lớn thì độ dốc càng gần 0 hơn. Vì lý do này nên gradient descent sẽ hội tụ chậm. Còn đối với hàm ReLU, với giá trị z dương thì độ dốc của hàm không gần bằng 0 như hàm tanh. Điều này giúp cho việc hội tụ xảy ra nhanh hơn. Với giá trị z âm, độ dốc bằng 0, tuy nhiên, hầu hết các giá trị của các neural trong mạng thường có giá trị dương, nên trường hợp này ít khi xảy ra. ReLU huấn luyện nhanh hơn so với sigmoid cũng bởi lý do tương tự.

### Reducing overfitting

#### Overfitting là gì?

Khi bạn dạy một đứa trẻ từ 2-5 tuổi về việc cộng hai số, chúng sẽ học rất nhanh và trả lời đúng hầu hết các câu hỏi mà chúng ta đã dạy chúng. Tuy nhiên, chúng sẽ trả lời sai đối với những câu hỏi hơi lắc léo một chút (câu hỏi tương tự câu chúng ta đã dạy, nhưng thêm một xíu thông tin đòi hỏi trẻ phải suy nghĩ), hoặc các câu hỏi chưa được dạy. Lý do chúng trả lời sai những câu hỏi đó là khi trả lời những câu hỏi được dạy, chúng thường nhớ lại câu trả lời, chứ không thực sự hiểu câu hỏi. Cái này ở Việt Nam ta gọi là học vẹt.

Tương tự vậy, Neural network chính bản thân nó có khả năng học được những gì được dạy, tuy nhiên, nếu quá trình huấn luyện của bạn không tốt, mô hình có khả năng sẽ giống như những đứa trẻ trên kia, hồi tưởng lại những gì đã dạy cho chúng mà không hiểu bản chất. Và kết quả Neural Network sẽ hoạt động tốt trên tập huấn luyện ( nhưng chúng không rút ra được bản chất chính của vấn đề), và kết quả trên tập test tệ. Người ta gọi trường hợp trên là **overfitting**.

Nhóm nghiên cứu AlexNet sử dụng nhiều phương pháp khác nhau để giảm overfitting.

#### Data Augmentation

Việc sử dụng nhiều biến thể khác nhau của một bức hình có thể giúp ngăn mô hình không bị overfitting. Với việc sử dụng nhiều biến thể của 1 bức hình, bạn bắt ép mô hình không học vẹt dữ liệu. Có nhiều cách khác nhau để sinh ra dữ liệu mới dựa vào dữ liệu có sẵn. Một vài các mà nhóm AlexNet đã sử dụng là.

##### Data Augmentation by Mirroring

Ý tưởng của việc này là lấy ảnh trong gương của một bức hình. Nhìn vào ảnh bên dưới, bên trái là hình gốc của con mèo trong tập huấn luyện, bên phải là ảnh của con mèo khi thêm hiệu ứng hình qua gương (đơn giản là xoay qua trục y là được )
![Tái tạo ảnh sử dụng phản ảnh](/post_image/AlexNet-Data-Augmentation-Mirror-Image.jpg)

##### Data Augmentation by Random Crops

Việc lựa chọn vị trí ảnh gốc một cách ngẫu nhiên cũng giúp chúng ta có thêm một ảnh khác so với ảnh gốc ban đầu.

Nhóm tác giả của AlexNet rút trích ngẫu nhiên bức ảnh có kích thước 227x227 từ bức ảnh 256x256 ban đầu làm input dầu vào cho mô hình. Bằng cách này, chúng ta có thể tăng số lượng dữ liệu lên gấp 2048 lần bằng việc sử dụng cách này.

![radom select](/post_image/AlexNet-Data-Augmentation-Random-Crops.jpg)

Bốn bức ảnh được crop ngẫu nhiên ở trên thoạt nhìn có vẻ giống nhau, nhưng thực chất không phải như vậy. 

Với việc sử dụng Data Augmentation, chúng ta đang bố gắng dạy cho mô hình rằng với việc nhìn hình con mèo qua gương, nó vẫn là con mèo, hoặc hình hình con mèo ở bất kỳ góc độ nào thì nó vẫn là nó. 

#### Dropout

Với gần 60 triệu tham số trong tập huấn luyện, việc overfitting xảy ra là đều dễ hiểu. Các tác giả của AlexNet đã thực nghiệm nhiều cách nữa để giảm overfitting. Họ sử dụng một kỹ thuật gọi là dropout - kỹ thuật này được giới thiệu ở bài báo khách của G.E. Hintol vào năm 2012. Kỹ thuật này khá đơn giản, một neural sẽ có xác suất bị loại khỏi mô hình là 0.5. Khi một neural bị loại khỏi mô hình, nó sẽ không được tham qia vào quá trình lan truyền tiến hoặc lan truyền ngược. Cho nên mỗi giá trị input sẽ đi qua một kiến trúc mạng khác nhau. Như mô tả ở hình động ở dưới, kết quả là giá trị của tham số trọng số sẽ tốt hơn và khó bị overfitting hơn. Trong quá trình test, toàn bộ network được sử dụng, không có dropout, tuy nhiên, giá trị output sẽ scaled bởi tham số 0.5 tương ứng với những neural không sử dụng trong quá trình trainning. Với việc sử dụng dropout, chúng ta sẽ tăng gấp đôi lần lặp cần thiết để đạt được độ hội tụ, nhưng khi không sử dụng dropout, mạng AlexNet rất dễ bị overfitting.

![drop out](/post_image/dropoutAnimation.gif)

Ngày nay, chuẩn hoá dropout là một yếu tố không thể thiếu và các mô hình sử dụng nó thường có kết quả tốt hơn so với mô hình tương tự không sử dụng dropout. Chúng ta sẽ bàn sâu hơn về dropout ở một bài khác trong tương lai.

Tham khảo

ImageNet Classification with Deep Convolutional Neural Networks  by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, 2012










