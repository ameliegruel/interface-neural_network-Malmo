from LM_model import *
import random

nb_tests = 10
nb_of_figures = 3
w = 0.025
# modif = 5
stimulation = 40

list_figures = []

file_results = open("./results_automaticLMmodel/results_automaticLMmodel.csv","w")
file_results.write("XXX test pour A dans XXX et PN_KC_w dans XXX, chaque combinaison pendant "+str(nb_tests)+" tests;;;;\n;;;;\nParameters;;;Learning rule;\nmin_weight;0,0001;;tc_et;40\nPN thresh;-40;;tc_plus;15\nPN rest;-60;;tc_minus;15\nKC rest;-85;;tc_reward;20\nEN thresh;-40;;reward;5\nEN rest;-60;;;\nPN-KC tc_synaptic;3;;;\nPN-KC phi;0,93;;;\nKC-EN tc_synaptic;8;;;\nKC-EN phi;8;;;\n;;;;\n;;;;\n")
file_results.write("A;PN KC weight;KC threshold;modification;stimulation;Number of correct guesses;Figures guessed correctly\n")

for (a,kc) in [(0.2,-10),(0.75,-10),(0.5,-11),(2.5,-11),(0.2,-12),(0.75,-12),(0.75,-13),(1.0,-13),(2.5,-13),(1.0,-14),(2.5,-14)] :

    for modif in [5,7]:
    # for stimulation in [40,50]:

        print("\n###################################\nA =",a, "and PN_KC_weight =", w, "and KC threshold =",kc,"and modification =",modif,"\n#################################\n")
        # print("\n###################################\nA =",a, "and PN_KC_weight =", w,"\n#################################\n")

        file_results.write(str(a)+";"+str(w)+";"+str(kc)+";"+str(modif)+";"+str(stimulation)+";")

        correct_guesses = {}
        how_many_times_guessed = {}

        for i in range(nb_tests):
            print("\n### Test",i,"###")

            if len(list_figures) != nb_tests:
                # figures = random.sample(range(1,11),nb_of_figures)
                # figures = random.sample([2,5,6,7,8], nb_of_figures-1)
                # figures.insert(0, figures[random.choice(range(nb_of_figures))])
                figures = random.sample([2,5,6,7,8], nb_of_figures-1)
                figures.insert(random.choice(range(nb_of_figures)), random.choice([1,3,4,9,10]))
                figures.insert(0, figures[random.choice(range(nb_of_figures))])
                figures = list(map(lambda x: "figure"+str(x)+".txt", figures))
                list_figures.append(figures)
            else : 
                figures = list_figures[i]

            result, similar_spikes = LM_model(figures=figures, info_PN=True, A=a, PN_KC_weight=w, KC_thresh=kc, modification=modif, stimulation_time=stimulation)

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