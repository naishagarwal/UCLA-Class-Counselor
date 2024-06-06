import json
import random

# prompt data
with open('prompts_classes.json', 'r') as fp:
    prompt_data = json.load(fp)

 
''' ### create random sample of prompts
# initializing the value of n
n = 1000
 
# printing n elements from list
prompt_list = random.sample(prompt_data, n)


with open('random_prompts.json', 'w') as fp:
    json.dump(prompt_list, fp)
print(prompt_list)
'''

# open randomized prompts
with open('random_prompts.json', 'r') as fp:
    prompt_list = json.load(fp)

# choose knn and prompt size
knn_size = 5
prompt_size = 100
prompt_list = prompt_list[750:750+prompt_size]

# class data
with open('cleaned_ucla_class_info.json', 'r') as fp:
    class_data = json.load(fp)

# results from embedding with different k values
with open('knn_{}.json'.format(knn_size), 'r') as fp:
    results_list = json.load(fp)


# check if class_name is in both result and prompt list
correct_counter = 0
final_list = []

for i, class_ in enumerate(class_data): 
    breakpoint = False
    for z, result in enumerate(results_list):
        for j, prompt in enumerate(prompt_list):
            if (class_data[i]['class_name'].split('Lec')[0].strip().lower() in str(result[str(z)]).lower() and
                class_data[i]['class_name'].split('Lec')[0].strip().lower() in prompt_list[j]['user'].lower()):
                print(str(result[str(z)]))
                if str(result[str(z)]) not in final_list:
                    correct_counter += 1
                    print(correct_counter)
                final_list.append(results_list[z][str(z)])
                breakpoint = True
                break
            else:
                continue
        if breakpoint is True:
            break
        else:
            continue

# print results
print(correct_counter)
print('accuracy score for {} random documents for knn = {} is: {}%'.format(prompt_size, knn_size, correct_counter/len(prompt_list)*100))