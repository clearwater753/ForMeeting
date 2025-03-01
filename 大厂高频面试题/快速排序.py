# 简短的快速排序实现
def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2] # 选择中间的数作为基准
    left = [x for x in nums if x < pivot]
    middle = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

if __name__ == "__main__":
    arr = [10, 7, 8, 9, 1, 5]
    sorted_arr = quick_sort(arr)
    print(sorted_arr)

# 快速排序
def quick_sort(nums, left, right):
    if left >= right:
        return
    pivot = partition(nums, left, right)
    quick_sort(nums, left, pivot - 1)
    quick_sort(nums, pivot + 1, right)

def partition(nums, left, right):
    pivot = nums[left]
    i, j = left, right
    while i < j:
        while i < j and nums[j] >= pivot:
            j -= 1
        while i < j and nums[i] <= pivot:
            i += 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[left], nums[i] = nums[i], nums[left]
    return i

# if __name__ == "__main__":
#     arr = [10, 7, 8, 9, 1, 5]
#     quick_sort(arr, 0, len(arr) - 1)
#     print(arr)
