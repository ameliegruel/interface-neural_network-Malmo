from LM_model import *
import random

file_results = open("results_automaticLMmodel.txt","w")

# for (a,w,kc) in [(0.75, 0.1, -20), (0.75, 0.1, -25), (1.0, 0.05, -30), (1.0, 0.05, -35), (1.0, 0.05, -45), (5.0, 0.1, -15),(0.5, 0.1, -25), (0.5, 0.05, -20), (0.5, 0.05, -25), (0.5, 0.05, -45), (0.5, 0.025, -40), (1.0, 0.05, -15), (1.0, 0.05, -20), (5.0, 0.1, -25), (5.0, 0.1, -35)]:

for (a,w) in [(0.5, 0.1), (0.5, 0.05), (0.5, 0.025), (0.75,0.1), (1.0,0.1), (1.0,0.05), (5.0, 0.1), (5.0, 0.05)]:

    for kc in [-15.0, -20.0, -25.0, -30.0,-35.0, -40.0, -45.0, -50.0]:

        file_results.write("### Test for A="+str(a)+" and PN_KC_weight="+str(w)+" and KC threshold="+str(kc)+" ###\n")
        print("\n###################################\nA =",a, "and PN_KC_weight =", w, "and KC threshold =",kc,"\n#################################\n")

        # file_results.write("##### BATCH "+str(i)+" #####")
        # print("\n###################################\nBATCH =",i,"\n#################################\n")

        # print("\n###################################\nA =",a, "and PN_KC_weight =", w,"\n#################################\n")
        # file_results.write("### Test for A="+str(a)+" and PN_KC_weight="+str(w)+" ###\n")

        # print("\n###################################\nKC threshold =", kc,"\n#################################\n")
        # file_results.write("### Test for KC threshold ="+str(kc)+" ###\n")

        correct_guesses = {}
        how_many_times_guessed = {}

        for i in range(50):
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
        file_results.write("Total results : "+ str(len(correct_guesses)) + " correct guess\n")
        file_results.write("Figures guessed correctly : ")
        for figure in correct_guesses.values():
            file_results.write(figure+" - ")

        # max_guessed = {"figure": None, "times": 0}
        # for (figure, times) in how_many_times_guessed.items() :
        #     if times > max_guessed["times"]:
        #         max_guessed["times"] = times
        #         max_guessed["figure"] = figure
        # print("\n\nFigure guessed the most :", max_guessed["figure"], "guessed", max_guessed["times"],"times\n",how_many_times_guessed)

        file_results.write("\n\n")

file_results.close()