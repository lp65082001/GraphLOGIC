from os import path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.ticker import FormatStrFormatter
#sns.set_theme()

class Plotter():

    def __init__(self, save_dir='.'):

        self.save_dir = save_dir

    def plot_pr(self, precision, recall, thres_list, name=None,
                filename_suffix=None):

        fig = plt.figure('pr', figsize=(4,4), dpi=300, constrained_layout=True)
        ax = fig.gca()

        # f1 contour
        levels = 10
        spacing = np.linspace(0, 1, 1000)
        x, y = np.meshgrid(spacing, spacing)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 / (1/x + 1/y)
        locx = np.linspace(0, 1, levels, endpoint=False)[1:]
        cs = ax.contour(x, y, f1, levels=levels, linewidths=1, colors='k',
                        alpha=0.3)
        ax.clabel(cs, inline=True, fmt='F1=%.1f',
                  manual=np.tile(locx,(2,1)).T)

        # compute f1_max and aupr
        with np.errstate(divide='ignore', invalid='ignore'):
            aupr = np.trapz(np.flip(precision), x=np.flip(recall))
            f1 = 2*recall*precision / (recall+precision)
        f1_max_idx = np.nanargmax(f1)
        f1_max = f1[f1_max_idx]

        ax.plot(recall, precision, lw=1, color='C0', label=name)

        ax.scatter(recall[f1_max_idx], precision[f1_max_idx],
                   label=f'{thres_list[f1_max_idx]:.4f} (f1$_{{max}}$)',
                   marker='o', edgecolors='C1',
                   facecolors='none', linewidths=0.5)
        if thres_list.size%2 == 1:
            def_thres_idx = (thres_list.size-1)//2
            ax.scatter(recall[def_thres_idx], precision[def_thres_idx],
                       label=f'{0.5:.4f}',
                       marker='x', c='C1',
                       linewidths=0.5)
        plt.legend(title='threshold', loc='lower left')

        plt.xlabel('recall',fontsize=12)
        plt.ylabel('precision',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.title(f'AUPR: {aupr}, f1: {f1_max}')
        plt.title(f'f1$_{{max}}$: {f1_max:.6f}',fontsize=12)

        # plt.show()
        filename = (f'pr_curve-{filename_suffix}.png'
                        if filename_suffix else 'pr_curve.png')
        plt.savefig(path.join(self.save_dir, filename))

        plt.close()
    def plot_roc(self, fpr,tpr,AUC, name=None,
                filename_suffix=None):

        fig = plt.figure('roc', figsize=(5,5), dpi=300, constrained_layout=True)
        ax = fig.gca()
   
        ax.plot(fpr, tpr, lw=1, color='C0')
        ax.plot([0,1],[0,1],"r--", lw=1)

        #plt.legend(title='threshold', loc='lower left')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # plt.title(f'AUPR: {aupr}, f1: {f1_max}')
        plt.title(f'AUC: {AUC:.6f}')

        # plt.show()
        filename = (f'roc_curve-{filename_suffix}.png'
                        if filename_suffix else 'roc_curve.png')
        plt.savefig(path.join(self.save_dir, filename))

        plt.close()
    def plot_cam(self, res, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(4,4)
        fig, ax1= plt.subplots(1)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        resid_cam=(res[0,:,0]).reshape(3,27)
        #print(resid_cam[0,:])
        ax1.plot(range(0,27),resid_cam[0,:],'bo-',alpha=0.7,label= "α1")
        ax1.plot(range(0,27),resid_cam[2,:],'go-',alpha=0.7,label= "α1")
        ax1.plot(range(0,27),resid_cam[1,:],'ro-',alpha=0.7,label= "α2")
        ax1.plot(12,0,color='orange',marker="*",markersize=10,markeredgecolor="black",label="mutation")
        #ax1.set_xlabel("GXY",fontsize=12)
        ax1.set_ylabel("Grad-CAM",fontsize=12)
        ax1.legend(loc="upper right",edgecolor="black",fontsize=12)
        ax1.set_xlim(-1,27)
        ax1.set_title(f"{name}, Lethal")
        plt.sca(ax1)
        plt.yticks(fontsize=12)
        plt.xticks(range(27),["G","P11","P10","G","P8","P7","G","P5","P4","G","P2","P1","G","P1'","P2'","G","P4'","P5'","G","P7'","P8'","G","P10'","P11'","G","P13'","P14'"],rotation=90,fontsize=11)
        
        fig.tight_layout()
        
        filename = (f'Grad-CAM-{name}.png'
                        if name else 'Grad-CAM.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300,bbox_inches="tight")
    
    def plot_cm(self, y_test,pred, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,6)
        fig, (ax1, ax2) = plt.subplots(1,2)
        cm = confusion_matrix(y_test,pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Lethal','Non-lethal'])
        disp.plot()
        filename = (f'Grad-CAM-{filename_suffix}.png'
                        if filename_suffix else 'cm.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300)

    def plot_classification(self, data1,data2, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,6)
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(data1.T, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax2.imshow(data2.T, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax1.set_title(f"{name} (a1)")
        ax2.set_title(f"{name} (a2)")
        #ax1.set_ylabel("a1")
        #ax2.set_ylabel("a2")
        ax2.set_xlabel("Position")

        plt.sca(ax1)
        plt.yticks(range(5), ["Pro","Small","Charged","Hydrophobic","Polar"])
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"])
        plt.sca(ax2)
        plt.yticks(range(5), ["Pro","Small","Charged","Hydrophobic","Polar"])
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"])
        fig.tight_layout()
        filename = (f'feature_heat_{name}.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300)
        plt.close()
    def plot_classification2(self, data1,data2, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,6)
        fig, (ax1, ax2) = plt.subplots(2,1)

        #ax1.matshow(np.argmax(data1.T,axis=0), cmap='Reds', interpolation='none', vmin=0, vmax=1)
        #ax2.matshow(np.argmax(data2.T,axis=0), cmap='Reds', interpolation='none', vmin=0, vmax=1)
        noa1 = np.where(np.sum(data1.T,axis=0)==0)[0]
        noa2 = np.where(np.sum(data2.T,axis=0)==0)[0]
        maxa1 = np.argmax(data1.T,axis=0)
        maxa2 = np.argmax(data2.T,axis=0)
        maxa1[noa1] = 5
        maxa2[noa2] = 5

        ax1.plot(range(18),maxa1,"*")
        ax2.plot(range(18),maxa2,"*")
        ax1.set_title(f"{name} (a1)")
        ax2.set_title(f"{name} (a2)")
        #ax1.set_ylabel("a1")
        #ax2.set_ylabel("a2")
        ax2.set_xlabel("Position")
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        plt.sca(ax1)
        plt.yticks(range(6), ["Pro","Small","Charged","Hydrophobic","Polar","Non"])
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"])
        plt.sca(ax2)
        plt.yticks(range(6), ["Pro","Small","Charged","Hydrophobic","Polar","Non"])
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"])
        fig.tight_layout()
        filename = (f'feature_heat_{name}_2.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300)
        plt.close()
    def plot_classification3(self, data1,data2,threshold=0.05, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,6)
        #fig, (ax1, ax2) = plt.subplots(2,1)
        xaxis1 = []
        yaxis1 = []
        xaxis2 = []
        yaxis2 = []
        for i in range(0,18):
            t_c = np.where(data1.T[:,i]>=threshold)[0]
            if (len(t_c)==0):
                #xaxis1.append(i-0.1)
                #yaxis1.append(5)
                pass
            else:
                for j in t_c:
                    xaxis1.append(i-0.1)
                    yaxis1.append(j)
        for i in range(0,18):
            t_c = np.where(data2.T[:,i]>=threshold)[0]
            if (len(t_c)==0):
                #xaxis2.append(i+0.1)
                #yaxis2.append(5)
                pass
            else:
                for j in t_c:
                    xaxis2.append(i+0.1)
                    yaxis2.append(j)
        plt.plot(xaxis1,yaxis1,"b^",label="α1")        
        plt.plot(xaxis2,yaxis2,"r*",label="α2")  
        plt.gca().invert_yaxis()
        plt.title(f"{name}, threshold:{int(threshold*100)}%")
        plt.yticks(range(5), ["Pro","Small","Charged","Hydrophobic","Polar"])
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"])
        plt.grid()
        plt.legend(bbox_to_anchor=(0.92,0.85),loc="center",edgecolor="black")
        filename = (f'feature_heat_{name}_3.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300)
        plt.close()
    def cal_intersect(self,d1,d2,d3,d4,threshold):
        xaxis1 = []
        yaxis1 = []
        xaxis2 = []
        yaxis2 = []
        xaxis3 = []
        yaxis3 = []
        xaxis4 = []
        yaxis4 = []
        for i in range(0,18):
            t_c = np.where(d1.T[:,i]>=threshold)[0]
            if (len(t_c)==0):
                #xaxis1.append(i-0.1)
                #yaxis1.append(5)
                pass
            else:
                for j in t_c:
                    xaxis1.append(i-0.1)
                    yaxis1.append(j)
        for i in range(0,18):
            t_c = np.where(d2.T[:,i]>=threshold)[0]
            if (len(t_c)==0):
                #xaxis2.append(i+0.1)
                #yaxis2.append(5)
                pass
            else:
                for j in t_c:
                    xaxis2.append(i+0.1)
                    yaxis2.append(j)
        for i in range(0,18):
            t_c = np.where(d3.T[:,i]>=threshold)[0]
            if (len(t_c)==0):
                #xaxis1.append(i-0.1)
                #yaxis1.append(5)
                pass
            else:
                for j in t_c:
                    xaxis3.append(i-0.1)
                    yaxis3.append(j)
        for i in range(0,18):
            t_c = np.where(d4.T[:,i]>=threshold)[0]
            if (len(t_c)==0):
                #xaxis2.append(i+0.1)
                #yaxis2.append(5)
                pass
            else:
                for j in t_c:
                    xaxis4.append(i+0.1)
                    yaxis4.append(j)
        xaxis1 = np.array(xaxis1)
        yaxis1 = np.array(yaxis1)
        xaxis2 = np.array(xaxis2)
        yaxis2 = np.array(yaxis2)
        xaxis3 = np.array(xaxis3)
        yaxis3 = np.array(yaxis3)
        xaxis4 = np.array(xaxis4)
        yaxis4 = np.array(yaxis4)
        la1 = np.vstack((xaxis1,yaxis1)).T
        la2 = np.vstack((xaxis2,yaxis2)).T
        #ls = np.vstack((la1,la2))
        nla1 = np.vstack((xaxis3,yaxis3)).T
        nla2 = np.vstack((xaxis4,yaxis4)).T
        #nls = np.vstack((nla1,nla2))

        sample1_a1 = np.array(list(set(map(tuple, la1)) - set(map(tuple, nla1))))
        sample1_a2 = np.array(list(set(map(tuple, la2)) - set(map(tuple, nla2))))
        sample2_a1 = np.array(list(set(map(tuple, nla1)) - set(map(tuple, la1))))
        sample2_a2 = np.array(list(set(map(tuple, nla2)) - set(map(tuple, la2))))
        return sample1_a1, sample1_a2,sample2_a1, sample2_a2


    def plot_classification4(self, data1,data2,data3,data4, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,4)
        #fig, (ax1, ax2) = plt.subplots(2,1)
        #sample1_a1 ,sample1_a2,sample2_a1,sample2_a2 = self.cal_intersect(data1,data2,data3,data4,threshold)

        fig, ax = plt.subplots(2,2)
        minmin = 0
        maxmax = 1

        #print(np.max(data1[0]))
        #print(np.max(data2[0]))
        #print(np.max(data3[0]))
        #print(np.max(data4[0]))
        
        im1 = ax[0,0].imshow(data1[0].T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im2 = ax[0,1].imshow(data2[0].T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im3 = ax[1,0].imshow(data3[0].T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im4 = ax[1,1].imshow(data4[0].T,vmin=minmin, vmax=maxmax,cmap="Reds")

        #plt.tight_layout()
        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        ax[0,0].set_title(f"{name}, ɑ1, Lethal")
        ax[0,1].set_title(f"{name}, ɑ1, Non-Lethal")
        ax[1,0].set_title(f"{name}, ɑ2, Lethal")
        ax[1,1].set_title(f"{name}, ɑ2, Non-Lethal")


        plt.sca(ax[0,0])
        plt.yticks(range(6), ["Pro","Small","Charged","Hydrophobic","Polar","Total"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[0,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,0])
        plt.yticks(range(6), ["Pro","Small","Charged","Hydrophobic","Polar","Total"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)

        filename = (f'feature_heat_{name}_4.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300,bbox_inches="tight")
        plt.close()
    def plot_classification4_total(self, data1,data2,data3,data4, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,4)
        #fig, (ax1, ax2) = plt.subplots(2,1)
        #sample1_a1 ,sample1_a2,sample2_a1,sample2_a2 = self.cal_intersect(data1,data2,data3,data4,threshold)

        fig, ax = plt.subplots(2,2)
        minmin = 0
        maxmax = 1

        #print(np.max(data1[0]))
        #print(np.max(data2[0]))
        #print(np.max(data3[0]))
        #print(np.max(data4[0]))
        
        im1 = ax[0,0].imshow(data1.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im2 = ax[0,1].imshow(data2.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im3 = ax[1,0].imshow(data3.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im4 = ax[1,1].imshow(data4.T,vmin=minmin, vmax=maxmax,cmap="Reds")

        #plt.tight_layout()
        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        ax[0,0].set_title(f"{name}, ɑ1, Lethal")
        ax[0,1].set_title(f"{name}, ɑ1, Non-Lethal")
        ax[1,0].set_title(f"{name}, ɑ2, Lethal")
        ax[1,1].set_title(f"{name}, ɑ2, Non-Lethal")


        plt.sca(ax[0,0])
        plt.yticks(range(6), ["Pro","Small","Charged","Hydrophobic","Polar","Total"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[0,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,0])
        plt.yticks(range(6), ["Pro","Small","Charged","Hydrophobic","Polar","Total"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)

        filename = (f'feature_heat_{name}_4.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300,bbox_inches="tight")
        plt.close()
    def plot_classification_by_type(self, data1,data2,data3,data4, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(8,4)
        #fig, (ax1, ax2) = plt.subplots(2,1)
        #sample1_a1 ,sample1_a2,sample2_a1,sample2_a2 = self.cal_intersect(data1,data2,data3,data4,threshold)

        fig, ax = plt.subplots(2,2)
        minmin = 0
        maxmax = 1

        #print(np.max(data1[0]))
        #print(np.max(data2[0]))
        #print(np.max(data3[0]))
        #print(np.max(data4[0]))
        
        im1 = ax[0,0].imshow(data1.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im2 = ax[0,1].imshow(data2.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im3 = ax[1,0].imshow(data3.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im4 = ax[1,1].imshow(data4.T,vmin=minmin, vmax=maxmax,cmap="Reds")

        #plt.tight_layout()
        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        ax[0,0].set_title(f"ɑ1, Lethal")
        ax[0,1].set_title(f"ɑ1, Non-Lethal")
        ax[1,0].set_title(f"ɑ2, Lethal")
        ax[1,1].set_title(f"ɑ2, Non-Lethal")

        plt.suptitle(f"{name} heat map")
        plt.sca(ax[0,0])
        plt.yticks(range(7), ["Ala","Cys","Asp","Glu","Arg","Ser","Val"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[0,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,0])
        plt.yticks(range(7), ["Ala","Cys","Asp","Glu","Arg","Ser","Val"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)

        filename = (f'feature_heat_{name}_4.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300,bbox_inches="tight")
        plt.close()
    def plot_classification_by_total(self, data1,data2,data3,data4, name=None,
                    filename_suffix=None):

            plt.rcParams['figure.figsize']=(8,4)
            #fig, (ax1, ax2) = plt.subplots(2,1)
            #sample1_a1 ,sample1_a2,sample2_a1,sample2_a2 = self.cal_intersect(data1,data2,data3,data4,threshold)

            fig, ax = plt.subplots(2,2)
            minmin = 0
            maxmax = 1

            #print(np.max(data1[0]))
            #print(np.max(data2[0]))
            #print(np.max(data3[0]))
            #print(np.max(data4[0]))
            
            im1 = ax[0,0].imshow(data1.T,vmin=minmin, vmax=maxmax,cmap="Reds")
            im2 = ax[0,1].imshow(data2.T,vmin=minmin, vmax=maxmax,cmap="Reds")
            im3 = ax[1,0].imshow(data3.T,vmin=minmin, vmax=maxmax,cmap="Reds")
            im4 = ax[1,1].imshow(data4.T,vmin=minmin, vmax=maxmax,cmap="Reds")

            #plt.tight_layout()
            # add space for colour bar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
            fig.colorbar(im2, cax=cbar_ax)

            ax[0,0].set_title(f"ɑ1, Lethal")
            ax[0,1].set_title(f"ɑ1, Non-Lethal")
            ax[1,0].set_title(f"ɑ2, Lethal")
            ax[1,1].set_title(f"ɑ2, Non-Lethal")

            plt.suptitle(f"{name} heat map")
            plt.sca(ax[0,0])
            plt.yticks(range(1), ["Probability"])
            plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
            plt.sca(ax[0,1])
            plt.yticks([])
            plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
            plt.sca(ax[1,0])
            plt.yticks(range(1), ["Probability"])
            plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
            plt.sca(ax[1,1])
            plt.yticks([])
            plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)

            filename = (f'feature_heat_{name}_total.png')
            plt.savefig(path.join(self.save_dir, filename),dpi=300,bbox_inches="tight")
            plt.close()


    def plot_classification_single(self, data1,data2,data3,data4, name=None,
                filename_suffix=None):

        plt.rcParams['figure.figsize']=(6,4)
        #fig, (ax1, ax2) = plt.subplots(2,1)
        #sample1_a1 ,sample1_a2,sample2_a1,sample2_a2 = self.cal_intersect(data1,data2,data3,data4,threshold)

        fig, ax = plt.subplots(2,1)
        minmin = 0
        maxmax = 1

        #print(np.max(data1[0]))
        #print(np.max(data2[0]))
        #print(np.max(data3[0]))
        #print(np.max(data4[0]))

        data1c = data1+data2
        data2c = data3+data4
        
        im1 = ax[0].imshow(data1c.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        im2 = ax[1].imshow(data2c.T,vmin=minmin, vmax=maxmax,cmap="Reds")
        

        #plt.tight_layout()
        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        #ax[0,0].set_title(f"ɑ1, Lethal")
        #ax[0,1].set_title(f"ɑ1, Non-Lethal")

        '''
        plt.suptitle(f"{name} heat map")
        plt.sca(ax[0,0])
        plt.yticks(range(7), ["Ala","Cys","Asp","Glu","Arg","Ser","Val"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[0,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,0])
        plt.yticks(range(7), ["Ala","Cys","Asp","Glu","Arg","Ser","Val"])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        plt.sca(ax[1,1])
        plt.yticks([])
        plt.xticks(range(18),["P11","P10","P8","P7","P5","P4","P2","P1","P1'","P2'","P4'","P5'","P7'","P8'","P10'","P11'","P13'","P14'"],rotation=90)
        '''
        filename = (f'total_map.png')
        plt.savefig(path.join(self.save_dir, filename),dpi=300,bbox_inches="tight")
        plt.close()
        


    
