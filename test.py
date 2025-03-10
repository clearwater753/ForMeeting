def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[0]
    left = [x for x in nums[1:] if x < pivot]
    middle = [x for x in nums if x == pivot]
    right = [x for x in nums[1:] if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


print(quick_sort([3,6,8,10,1,2,1]))
