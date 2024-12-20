import math

def calculate_new_median(old_median, new_value, count, data_list):
    if count % 2 == 0:
        if new_value <= old_median and new_value <= data_list[(count // 2) - 1]:
            return data_list[(count // 2) - 1]
        elif (new_value <= old_median and new_value > data_list[(count // 2) - 1]) or (new_value > old_median and new_value <= data_list[count // 2]):
            return new_value
        else:
            return data_list[count // 2]
    else:
        if new_value <= old_median and new_value <= data_list[(count // 2) - 1]:
            return (data_list[(count // 2) - 1] + old_median) / 2
        elif (new_value <= old_median and new_value > data_list[(count // 2) - 1]) or (new_value > old_median and new_value <= data_list[(count // 2) + 1]):
            return (new_value + old_median) / 2
        else:
            return (data_list[(count // 2) + 1] + old_median) / 2

def calculate_new_std(old_mean, old_std, new_mean, new_value, count, data_list):
    return math.sqrt(((count * old_mean ** 2) + (count * old_std ** 2) + (new_value ** 2)) / (count + 1) - (new_mean ** 2))

# Example values
old_median = 5.0
new_value = 7.0
count = 10
data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

new_median = calculate_new_median(old_median, new_value, count, data_list)

old_mean = 5.5
old_std = 2.5
new_mean = (old_mean * count + new_value) / (count + 1)

new_std = calculate_new_std(old_mean, old_std, new_mean, new_value, count, data_list)

print(f"NewMean: {new_mean:.2f}")
print(f"NewMedian: {new_median:.2f}")
print(f"NewStd: {new_std:.2f}")