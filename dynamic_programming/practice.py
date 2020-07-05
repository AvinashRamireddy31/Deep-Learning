# https://www.geeksforgeeks.org/minimum-number-of-distinct-elements-after-removing-m-items/

def value_counts(items):
    dict_map = {}

    for i in items:
        if i in dict_map:
            dict_map[i] = dict_map[i] + 1
        else:
            dict_map[i] = 1

    return dict_map

def get_sorted_value_counts(dict_data):
    return {k:v for k,v in sorted(dict_data.items(), key = lambda i:i[1], reverse = False)}


# dict_data = value_counts([1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
dict_data = value_counts([2, 2, 1, 3, 3, 3])

sorted_dict_data = get_sorted_value_counts(dict_data)

def get_output(sorted_dict_data, remove_count):  
    for key in sorted_dict_data.keys():
        value = sorted_dict_data[key] 

        if remove_count >= value: 
            sorted_dict_data[key] = 0 # Remove all values(count) 
            
            remove_count -= value
            
            print("Inside", remove_count)
        else:  
            sorted_dict_data[key] = value - remove_count
            remove_count = 0 # Exhausted 

            print("Else", remove_count) 
 
    print(sorted_dict_data)

    remaining = [v for v in sorted_dict_data.values() if v > 0]

    return len(remaining)

#Given in question
remove_count = 3

output = get_output(sorted_dict_data, remove_count)
print("Unique :",output)






