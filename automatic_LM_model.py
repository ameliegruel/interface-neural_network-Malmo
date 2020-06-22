from LM_model import *
import random

file_results = open("./results_automaticLMmodel/results_automaticLMmodel.csv","w")
kc = -25

file_results.write("A;PN KC weight;KC threshold;Number of correct guesses;Figures guessed correctly\n")

for a in [0.2,0.5,0.75,1.0,2.5,5.0] :

    for w in [0.25,0.1,0.05,0.025,0.01,0.005]:

        # print("\n###################################\nA =",a, "and PN_KC_weight =", w, "and KC threshold =",kc,"\n#################################\n")
        print("\n###################################\nA =",a, "and PN_KC_weight =", w,"\n#################################\n")

        file_results.write(str(a)+";"+str(w)+";"+str(kc)+";")

        correct_guesses = {}
        how_many_times_guessed = {}

        for i in range(10):
            print("\n### Test",i,"###")
            nb_of_figures = 3
            figures = random.sample(range(1,11),nb_of_figures)
            figures.insert(0, figures[random.choice(range(nb_of_figures))])
            figures = list(map(lambda x: "figure"+str(x)+".txt", figures))
            result, similar_spikes = LM_model(figures=figures, A=a, PN_KC_weight=w, KC_thresh=kc)

            if result == figures[0] and similar_spikes == False:
                correct_guesses["Test "+str(i)] = result

            # if result not in how_many_times_guessed.keys():
            #     how_many_times_guessed[result] = 1
            # else : 
            #     how_many_times_guessed[result] += 1

        print("\n\nTotal results :", len(correct_guesses), "correct guess\n",correct_guesses)
        file_results.write(str(len(correct_guesses))+ ";")
        for figure in correct_guesses.values():
            file_results.write(figure+" - ")

        # max_guessed = {"figure": None, "times": 0}
        # for (figure, times) in how_many_times_guessed.items() :
        #     if times > max_guessed["times"]:
        #         max_guessed["times"] = times
        #         max_guessed["figure"] = figure
        # print("\n\nFigure guessed the most :", max_guessed["figure"], "guessed", max_guessed["times"],"times\n",how_many_times_guessed)

        file_results.write("\n")

file_results.close()