# import hashlib

# class BloomFilter:
#     def __init__(self, size, num_hash_functions):
#         # Kích thước Bloom filter (số lượng bit)
#         self.size = size
#         # Số hàm băm
#         self.num_hash_functions = num_hash_functions
#         # Mảng bit để lưu trữ (khởi tạo với các bit = 0)
#         self.bit_array = [0] * size

#     def _hash(self, item, i):
#         """Hàm băm để tạo chỉ số từ item, với i là chỉ số của hàm băm."""
#         return int(hashlib.sha256(f"{item}{i}".encode('utf-8')).hexdigest(), 16) % self.size

#     def add(self, item):
#         """Thêm một item vào Bloom filter."""
#         for i in range(self.num_hash_functions):
#             index = self._hash(item, i)
#             self.bit_array[index] = 1

#     def check(self, item):
#         """Kiểm tra một item có trong Bloom filter không."""
#         for i in range(self.num_hash_functions):
#             index = self._hash(item, i)
#             if self.bit_array[index] == 0:
#                 return False  # Nếu bất kỳ bit nào là 0, phần tử chắc chắn không có trong bộ lọc
#         return True  # Nếu tất cả các bit đều là 1, có khả năng phần tử có trong bộ lọc

# # Khởi tạo Bloom filter với kích thước 1000 bit và 3 hàm băm
# bloom_filter = BloomFilter(size=1000, num_hash_functions=3)

# # Mô phỏng thông tin giao dịch của người dùng
# users_transactions = {
#     "tung": ["New York", "Los Angeles", "Miami"],
#     "kim": ["London", "Paris", "Berlin"],
#     "tuan": ["Tokyo", "Osaka", "Kyoto"]
# }

# # Thêm tất cả các giao dịch vào Bloom filter
# for user, locations in users_transactions.items():
#     for location in locations:
#         bloom_filter.add(f"{user}-{location}")  # Kết hợp tên người dùng và địa điểm giao dịch

# # Kiểm tra một giao dịch mới từ người dùng
# def check_fraud(user, location):
#     if not bloom_filter.check(f"{user}-{location}"):
#         return f"Warning: Transaction from {location} by {user} might be suspicious!"
#     else:
#         return f"Transaction from {location} by {user} is normal."

# # Kiểm tra một số giao dịch
# test_transactions = [
#     ("tung", "Miami"),  # Giao dịch hợp lệ
#     ("tung", "Chicago"),  # Giao dịch mới (không có trong Bloom filter)
#     ("kim", "Paris"),  # Giao dịch hợp lệ
#     ("tuan", "Kyoto")   # Giao dịch hợp lệ
# ]

# # Kiểm tra các giao dịch
# for user, location in test_transactions:
#     result = check_fraud(user, location)
#     print(result)






# import hashlib

# class BloomFilter:
#     def __init__(self, size, num_hash_functions):
#         # Kích thước của Bloom filter (số lượng bit)
#         self.size = size
#         # Số hàm băm
#         self.num_hash_functions = num_hash_functions
#         # Mảng bit để lưu trữ (được khởi tạo với tất cả các bit là 0)
#         self.bit_array = [0] * size

#     def _hash(self, item, i):
#         """Hàm băm để tạo chỉ số từ item, với i là chỉ số của hàm băm."""
#         # Dùng hàm băm SHA-256 và điều chỉnh với chỉ số i để tạo ra các chỉ số khác nhau
#         return int(hashlib.sha256(f"{item}{i}".encode('utf-8')).hexdigest(), 16) % self.size

#     def add(self, item):
#         """Thêm một item vào Bloom filter."""
#         for i in range(self.num_hash_functions):
#             index = self._hash(item, i)
#             self.bit_array[index] = 1

#     def check(self, item):
#         """Kiểm tra một item có trong Bloom filter không."""
#         for i in range(self.num_hash_functions):
#             index = self._hash(item, i)
#             if self.bit_array[index] == 0:
#                 return False  # Nếu bất kỳ bit nào là 0, phần tử chắc chắn không có trong bộ lọc
#         return True  # Nếu tất cả các bit đều là 1, có khả năng phần tử có trong bộ lọc

# # Tạo một Bloom filter với kích thước 1000 bit và 3 hàm băm
# bloom_filter = BloomFilter(size=1000, num_hash_functions=3)

# # Từ điển mẫu
# dictionary = ["hello", "world", "spell", "check", "python", "bloom", "filter"]

# # Thêm tất cả các từ vào Bloom filter
# for word in dictionary:
#     bloom_filter.add(word)

# # Kiểm tra chính tả của một số từ
# test_words = ["hello", "world", "java", "python", "flutter"]

# for word in test_words:
#     if bloom_filter.check(word):
#         print(f"'{word}' có thể là một từ đúng.")
#     else:
#         print(f"'{word}' chắc chắn là một từ sai.")



import hashlib

class BloomFilter:
    def __init__(self, size, num_hash_functions):
        # Kích thước Bloom filter (số lượng bit)
        self.size = size
        # Số hàm băm
        self.num_hash_functions = num_hash_functions
        # Mảng bit để lưu trữ (khởi tạo tất cả các bit là 0)
        self.bit_array = [0] * size

    def _hash(self, item, i):
        """Hàm băm để tạo chỉ số từ item, với i là chỉ số của hàm băm."""
        return int(hashlib.sha256(f"{item}{i}".encode('utf-8')).hexdigest(), 16) % self.size

    def add(self, item):
        """Thêm một item vào Bloom filter."""
        for i in range(self.num_hash_functions):
            index = self._hash(item, i)
            self.bit_array[index] = 1

    def check(self, item):
        """Kiểm tra một item có trong Bloom filter không."""
        for i in range(self.num_hash_functions):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False  # Nếu bất kỳ bit nào là 0, phần tử chắc chắn không có trong bộ lọc
        return True  # Nếu tất cả các bit đều là 1, có khả năng phần tử có trong bộ lọc

# Khởi tạo Bloom filter
bloom_filter = BloomFilter(size=1000, num_hash_functions=3)

# Mô phỏng danh sách sản phẩm người dùng đã tương tác
user_interactions = {
    "tung": ["thịt bò", "hành tây", "khoai tây"],
    "tuan": ["thit heo", "trứng"],
    "canh": ["thịt bò", "trứng", "sữa TH", "kem"]
}

# Thêm các sản phẩm đã tương tác vào Bloom filter
for user, items in user_interactions.items():
    for item in items:
        bloom_filter.add(f"{user}-{item}")  # Kết hợp user và item để lưu trữ duy nhất

# Hàm gợi ý sản phẩm
def recommend_items(user, candidate_items):
    """Đưa ra gợi ý các sản phẩm chưa tương tác."""
    recommendations = []
    for item in candidate_items:
        if not bloom_filter.check(f"{user}-{item}"):
            recommendations.append(item)  # Chỉ thêm sản phẩm nếu chưa tương tác
    return recommendations

# Danh sách các sản phẩm có thể gợi ý
candidate_items = ["thịt bò", "hành tây", "khoai tây","trứng", "sữa TH", "kem","thit heo"]

# Gợi ý sản phẩm cho user1
user = "tung"
recommendations = recommend_items(user, candidate_items)

print(f"Recommendations for {user}: {recommendations}")
