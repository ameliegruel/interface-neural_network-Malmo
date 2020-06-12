from LM_model import *
import random

correct_guesses = {}
how_many_times_guessed = {}

for i in range(50):
    print("\n### Test",i,"###")
    nb_of_figures = 3
    figures = random.sample(range(1,11),nb_of_figures)
    figures.insert(0, figures[random.choice(range(nb_of_figures))])
    figures = list(map(lambda x: "figure"+str(x)+".txt", figures))
    result = LM_model(figures=figures)

    if result == figures[0]:
        correct_guesses["Test "+str(i)] = result

    if result not in how_many_times_guessed.keys():
        how_many_times_guessed[result] = 1
    else : 
        how_many_times_guessed[result] += 1

print("\n\nTotal results :", len(correct_guesses), "correct guess\n",correct_guesses)

max_guessed = {"figure": None, "times": 0}
for (figure, times) in how_many_times_guessed.items() :
    if times > max_guessed["times"]:
        max_guessed["times"] = times
        max_guessed["figure"] = figure
print("\n\nFigure guessed the most :", max_guessed["figure"], "guessed", max_guessed["times"],"times\n",how_many_times_guessed)
