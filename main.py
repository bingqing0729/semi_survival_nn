from models.simulator import SimulatedSurvival_V
import matplotlib.pyplot as plt
import numpy as np
from models.g_deep import g_D
from models.beta_est import Beta_est
from sklearn.preprocessing import StandardScaler

n = 500
n_test = 10
step = 0.1
N_sim = 1
generator = SimulatedSurvival_V(100,step)

x_test,loc_test,w_test,z_test,dt_test,fail_test,ind_test,t_test,cum_g_true_list,hazard_true_list = generator.generate_data(n_test,test=True,seed=0)
x_test = np.stack((x_test,loc_test,w_test),-1)

beta_list = []
cum_g_pred_list =[]
hazard_pred_list = []

for i in range(N_sim):

    x_train,loc_train,w_train,z_train,dt_train,fail_train,ind_train = generator.generate_data(n,seed=i+1)
    x_train = np.stack((x_train,loc_train,w_train),-1)
    placeholder = np.zeros(ind_train.shape)
    x_valid,loc_valid,w_valid,z_valid,dt_valid,fail_valid,ind_valid = generator.generate_data(n,seed=N_sim+i+1)
    x_valid = np.stack((x_valid,loc_valid,w_valid),-1) 

    beta = [0,0]
    prev_loss = np.inf
    nn_config = {
        "hidden_layers_nodes": 8,
        "learning_rate": 0.001,
        "activation": 'relu',
        "optimizer": 'adam', 
        "batch_size": 500,
        "patience": 10
        }

    

    for _ in range(20):
        min_loss = np.inf
        for _ in range(1):
            model_temp = g_D(nn_config)
            loss = model_temp.train(
                x_train = x_train,
                z_train = z_train,
                dt_train = dt_train,
                fail_train = fail_train,
                ind_train = ind_train,
                placeholder_train = placeholder,
                x_valid = x_valid,
                z_valid = z_valid,
                dt_valid = dt_valid,
                fail_valid = fail_valid,
                ind_valid = ind_valid,
                placeholder_valid = placeholder,
                beta = beta
                )
            
            if loss < min_loss:
                model = model_temp
                min_loss = loss


        train_pred = model.model.predict((x_train,z_train,dt_train,ind_train,fail_train))
        cum_g_pred = np.squeeze(train_pred)

        valid_pred = model.model.predict((x_valid,z_valid,dt_valid,ind_valid,fail_valid))
        cum_g_pred_v = np.squeeze(valid_pred)

        new_beta = Beta_est(ind_train[:,:,0],fail_train[:,:,0],z_train,dt_train[:,:,0],cum_g_pred,
                            ind_valid[:,:,0],fail_valid[:,:,0],z_valid,dt_valid[:,:,0],cum_g_pred_v)
        print("beta: ", new_beta)
        beta = new_beta
        with open('results/beta.csv','a') as file:
            file.write(','.join(map(str,new_beta))+', loss: '+str(loss)+'\n')
    
    test_pred = model.model.predict((x_test,z_test,dt_test,ind_test,fail_test))
    cum_g_pred = np.squeeze(test_pred)
    beta_list.append(beta)
    cum_g_pred_list.append(cum_g_pred)

    hazard_pred = np.exp(beta[0]+beta[1]*z_test+cum_g_pred)
    hazard_pred_list.append(hazard_pred)

beta_all = list(zip(*beta_list))
plt.boxplot(beta_all[0])
plt.savefig("results/beta_0.png")
plt.close()
plt.boxplot(beta_all[1])
plt.savefig("results/beta_1.png")
plt.close()


for j in range(n_test):
    plt.plot(t_test,np.mean(cum_g_pred_list,axis=0)[j],label="pred",color='green')
    plt.plot(t_test,cum_g_true_list[j],label="truth",color='black')
    plt.plot(t_test,np.percentile(cum_g_pred_list,5,axis=0)[j],color='green',linestyle='dashed')
    plt.plot(t_test,np.percentile(cum_g_pred_list,95,axis=0)[j],color='green',linestyle='dashed')
    plt.savefig("results/figure_cum_g_{}.png".format(j))
    plt.close()
    plt.plot(t_test,np.mean(hazard_pred_list,axis=0)[j],label="pred",color='green')
    plt.plot(t_test,hazard_true_list[j],label="truth",color='black')
    plt.plot(t_test,np.percentile(hazard_pred_list,5,axis=0)[j],color='green',linestyle='dashed')
    plt.plot(t_test,np.percentile(hazard_pred_list,95,axis=0)[j],color='green',linestyle='dashed')
    plt.savefig("results/figure_{}.png".format(j))
    plt.close()