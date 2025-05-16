import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.
        """
        x = abs(np.mean(self.genuine_scores) - np.mean(self.impostor_scores))
        y = np.sqrt(0.5 * (np.var(self.genuine_scores) + np.var(self.impostor_scores)))
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure()
        
        plt.hist(self.genuine_scores, bins=30, color='green', alpha=0.5, label='Genuine Scores', lw=2, histtype='step', hatch='/')
        plt.hist(self.impostor_scores, bins=30, color='red', alpha=0.5, label='Impostor Scores', lw=2, histtype='step', hatch='\\')
        plt.xlim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left', fontsize=10)
        plt.xlabel('Score', fontsize=12, weight='bold')
        plt.ylabel('Frequency', fontsize=12, weight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % (self.get_dprime(), self.plot_title), fontsize=15, weight='bold')
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
        """
        eer_diff = np.abs(FPR - FNR)
        min_index = np.argmin(eer_diff)
        return (FPR[min_index] + FNR[min_index]) / 2

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        """
        EER = self.get_EER(FPR, FNR)
        
        plt.figure()
        plt.plot(FPR, FNR, lw=2, color='blue', label='DET Curve')
        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('False Pos. Rate', fontsize=12, weight='bold')
        plt.ylabel('False Neg. Rate', fontsize=12, weight='bold')
        plt.title('Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s' % (EER, self.plot_title), fontsize=15, weight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig('det_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        """
        plt.figure()
        plt.plot(FPR, TPR, color='orange', lw=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
        plt.title('ROC Curve\nSystem %s' % self.plot_title, fontsize=15, weight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig('roc_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def compute_rates(self):
        """
        Calculate FPR, FNR, and TPR based on thresholds.
        """
        FPR, FNR, TPR = [], [], []
        for threshold in self.thresholds:
            TP = np.sum(self.genuine_scores >= threshold)
            FP = np.sum(self.impostor_scores >= threshold)
            TN = np.sum(self.impostor_scores < threshold)
            FN = np.sum(self.genuine_scores < threshold)
            
            FPR.append(FP / (FP + TN + self.epsilon))
            FNR.append(FN / (TP + FN + self.epsilon))
            TPR.append(TP / (TP + FN + self.epsilon))
        
        return np.array(FPR), np.array(FNR), np.array(TPR)


def main():
    # Set the random seed to 1.
    np.random.seed(1)

    systems = ["A", "B", "C"]

    for system in systems:
        genuine_mean = np.random.uniform(0.5, 0.9)
        genuine_std = np.random.uniform(0.0, 0.2)
        genuine_scores = np.random.normal(genuine_mean, genuine_std, 400)
        
        impostor_mean = np.random.uniform(0.1, 0.5)
        impostor_std = np.random.uniform(0.0, 0.2)
        impostor_scores = np.random.normal(impostor_mean, impostor_std, 1600)
        
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=200,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title="System %s" % system
        )
        
        FPR, FNR, TPR = evaluator.compute_rates()
        evaluator.plot_score_distribution()
        evaluator.plot_det_curve(FPR, FNR)
        evaluator.plot_roc_curve(FPR, TPR)


if __name__ == "__main__":
    main()
