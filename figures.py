import matplotlib.pyplot as plt
import numpy as np

eps = 0.1
coef = 50
coefs_rational = [10,50,200,1000]
coefs_sigmoid = [5,10,30,200]
f_type = 'rationals' # rational, binary, sigmod, all
lim = 0.2



# 다크 블루 그라데이션 팔레트 생성
def blue_gradient_palette(n):
    return [(i / (n-1) * 0.8, i / (n-1) * 0.8, 1.0) for i in range(n)]
# 다크 그린 그라데이션 팔레트 생성
def green_gradient_palette(n):
    return [(0, (i / (n-1)), 0) for i in range(n)]



# 유리함수 정의
def rational_f(x, coef):
    return (1 - 1 / (coef * x + 1))

# 시그모이드 함수 정의
def sigmoid_f(x, coef, eps):
    return (1 / (1 + np.exp(-coef * (x - eps))))

# binary 함수 정의
def binary_f(x):
    y = []
    for i in x:
        if i == 0:
            y.append(0)
        else:
            y.append(1)
    # print(y)
    return y

def binary_eps_f(x, eps):
    y = []
    for i in x:
        if i <= eps:
            y.append(0)
        else:
            y.append(1)
    # print(y)
    return y

# x 값 범위 설정 (비연속점을 제외)
x = np.linspace(0, 1, 400)

if f_type == 'rational':
    # 그래프 그리기
    plt.figure(figsize=(5,5))
    plt.plot(0,0,'bo', markersize=5)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    
    # plt.axhline(y=1, color='gray', linestyle='--', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.plot([0,1], [1,1], color='gray', linestyle='--', lw=1)
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    
    # plt.axvline(x=1, color='gray', linestyle='--', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.plot(x, rational_f(x, coef), label=r'$f(x) = 1 - \frac{1}{coef*w_i + 1}$', color='blue')
    # plt.xlabel(r'$w_i$')
    # plt.ylabel(r'$\tilde{b_i}(w_i)$')
    plt.xlim(0-lim, 1+lim)
    plt.ylim(0-lim, 1+lim)  # y 값 범위 설정
    plt.xticks([])
    plt.yticks([])
    for i in np.arange(0, 1.1, 0.2):
        if i==0:
            plt.text(i,-0.05,f'{i:.1f}', ha='center', va='center')
            continue
        plt.text(i, -0.05, f'{i:.1f}', ha='center', va='center')
        plt.text(-0.05, i, f'{i:.1f}', ha='center', va='center')


    # plt.title(r'$\tilde{b_i}(w_i)$')
    # plt.legend()
    # plt.grid(True)
    plt.show()
    plt.savefig(f"./figures/rational_f_{coef}.png")
    

elif f_type == 'sigmoid':
    # 그래프 그리기
    plt.figure(figsize=(5,5))
    # plt.plot(eps,0.5,'bo', markersize=5)
    plt.plot([eps,eps], [0,0.5], color='gray', linestyle='--', lw=1)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.plot([0,1], [1,1], color='gray', linestyle='--', lw=1)
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.plot(x, sigmoid_f(x, coef, eps), label=r'$f(x) = 1 - \frac{1}{coef*w_i + 1}$', color='blue')
    # plt.xlabel(r'$w_i$')
    # plt.ylabel(r'$\tilde{b_i}(w_i)$')
    plt.xlim(0-lim, 1+lim)
    plt.ylim(0-lim, 1+lim) 
    plt.xticks([])
    plt.yticks([])
    for i in np.arange(0, 1.1, 0.2):
        if i==0:
            plt.text(i,-0.05,f'{i:.1f}', ha='center', va='center')
            continue
        plt.text(i, -0.05, f'{i:.1f}', ha='center', va='center')
        plt.text(-0.05, i, f'{i:.1f}', ha='center', va='center')
    plt.text(eps,-0.05,r"$\epsilon$", ha='center',va='center')
    # plt.title(r'$\tilde{b_i}(w_i)$')
    # plt.legend()
    # plt.grid(True)
    plt.show()
    plt.savefig(f"./figures/sigmoid_f_{coef}.png")
    
    
elif f_type == 'binary':
    # 그래프 그리기
    plt.figure(figsize=(5,5))
    # plt.scatter(0, 1, s=30, edgecolors='blue', facecolors='none', linewidths=2)
    plt.plot(1,1,'bo', markersize=5)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.scatter(x, binary_f(x), color='blue', s=1)
    plt.plot(0,0,'bo', markersize=5)
    plt.plot(0, 1, 'o', markersize=5, markerfacecolor='none', markeredgecolor='blue',linewidth=3)
    # plt.xlabel(r'$w_i$')
    # plt.ylabel(r'$\tilde{b_i}(w_i)$')
    plt.xlim(0-lim, 1+lim)
    plt.ylim(0-lim, 1+lim) 
    plt.xticks([])
    plt.yticks([])
    for i in np.arange(0, 1.1, 0.2):
        if i==0:
            plt.text(i,-0.05,f'{i:.1f}', ha='center', va='center')
            continue
        plt.text(i, -0.05, f'{i:.1f}', ha='center', va='center')
        plt.text(-0.05, i, f'{i:.1f}', ha='center', va='center')
    # plt.title(r'$\tilde{b_i}(w_i)$')
    # plt.legend()
    # plt.grid(True)
    plt.show()
    plt.savefig(f"./figures/binary_f_.png")

elif f_type == 'rationals':
    # rational
    plt.figure(figsize=(5,5))
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    n_colors = len(coefs_rational)
    palette = green_gradient_palette(n_colors)
    palette = palette[::-1]
    label = [r'$a=10$',r'$a=50$',r'$a=200$',r'$a=1000$']
    for i,coe in enumerate(coefs_rational):
        plt.plot(x, rational_f(x, coe), label=label[i], color=palette[i])
        # plt.text(x, rational_f(x, coe), 'Local Max', ha='left', va='bottom', color=palette[i], fontsize=12)
    plt.legend(fontsize=13,loc='center', bbox_to_anchor=(0.8,0.4))
    # binary
    plt.plot(1,1,'ro', markersize=5)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.scatter(x, binary_f(x), color='red', s=1)
    plt.plot(0,0,'ro', markersize=5)
    plt.plot(0, 1, 'o', markersize=5, markerfacecolor='none', markeredgecolor='red',linewidth=3)
    
    # plt.xlabel(r'$w_i$')
    # plt.ylabel(r'$\tilde{b_i}(w_i)$')
    plt.xlim(0-lim, 1+lim)
    plt.ylim(0-lim, 1+lim)  # y 값 범위 설정
    plt.xticks([])
    plt.yticks([])
    for i in np.arange(0, 1.1, 0.2):
        if i==0:
            plt.text(i,-0.05,f'{i:.1f}', ha='center', va='center')
            continue
        plt.text(i, -0.05, f'{i:.1f}', ha='center', va='center')
        plt.text(-0.05, i, f'{i:.1f}', ha='center', va='center')
    plt.show()
    plt.savefig(f"./figures/rationals_f.png")

elif f_type == 'sigmoids':
    # sigmod
    plt.figure(figsize=(5,5))
    # plt.plot(eps,0.5,'bo', markersize=5)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.plot([eps,eps], [0,1], color='gray', linestyle='--', lw=1)
    n_colors = len(coefs_sigmoid)
    palette = blue_gradient_palette(n_colors)
    palette = palette[::-1]
    label = [r'$a=5$',r'$a=10$',r'$a=30$',r'$a=200$']
    for i,coe in enumerate(coefs_sigmoid):
        plt.plot(x, sigmoid_f(x, coe, eps), label=label[i], color=palette[i])
    plt.legend(fontsize=13,loc='center', bbox_to_anchor=(0.8,0.4))
    # binary
    plt.plot(1,1,'ro', markersize=5)
    plt.plot(eps,0,'ro', markersize=5)
    plt.axhline(y=0, color='black', linestyle='-', lw=1)#, label=r'Asymptote $\tilde{b_i}=1$')
    plt.axvline(x=0, color='black', linestyle='-', lw=1)#, label='Asymptote $w_i=1$')
    plt.plot([1,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.scatter(x, binary_eps_f(x,eps), color='red', s=1)
    plt.plot(0,0,'ro', markersize=5)
    plt.plot(eps, 1, 'o', markersize=5, markerfacecolor='none', markeredgecolor='red',linewidth=3)
    # plt.plot([0,eps], [0.5,0.5], color='gray', linestyle='--', lw=1)
    
    plt.xlim(0-lim, 1+lim)
    plt.ylim(0-lim, 1+lim) 
    plt.xticks([])
    plt.yticks([])
    for i in np.arange(0, 1.1, 0.2):
        if i==0:
            plt.text(i,-0.05,f'{i:.1f}', ha='center', va='center')
            continue
        plt.text(i, -0.05, f'{i:.1f}', ha='center', va='center')
        plt.text(-0.05, i, f'{i:.1f}', ha='center', va='center')
    plt.text(eps,-0.05,r"$\epsilon$", ha='center',va='center')
    
    plt.show()
    plt.savefig(f"./figures/sigmoids_f.png")
    
elif f_type == 'all':
    coefs = [50, 1000]
    coefs2 = [30, 100]
    x = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 5,figsize=(25,5))
    
    # binary 함수 그래프
    axes[0].plot(1,1,'bo',markersize=5)
    axes[0].plot(0,0,'bo', markersize=5)
    axes[0].plot(0, 1, 'o', markersize=5, markerfacecolor='none', markeredgecolor='blue',linewidth=3)
    axes[0].scatter(x, binary_f(x), color='blue', s=1)
    axes[0].set_title(f'Binary Function')
    axes[0].set_xlim(0-lim, 1+lim)
    axes[0].set_ylim(0-lim, 1+lim)

    # rational 함수 그래프
    axes[1].plot(0,0,'bo', markersize=5, color='green')
    axes[1].plot(x, rational_f(x, coefs[0]), color='green')
    axes[1].set_title(f'Rational Function with coef={coefs[0]}')
    axes[1].set_xlim(0-lim, 1+lim)
    axes[1].set_ylim(0-lim, 1+lim)
    
    axes[2].plot(0,0,'bo', markersize=5, color='green')
    axes[2].plot(x, rational_f(x, coefs[1]), color='green')
    axes[2].set_title(f'Rational Function with coef={coefs[1]}')
    axes[2].set_xlim(0-lim, 1+lim)
    axes[2].set_ylim(0-lim, 1+lim)


    # sigmoid 함수 그래프
    axes[3].plot(eps,0.5,'bo', markersize=5, color='red')
    axes[3].plot(x, sigmoid_f(x, coefs[0], eps), color='red')
    axes[3].set_title(f'Sigmoid Function with coef={coefs2[0]}')
    axes[3].set_xlim(0-lim, 1+lim)
    axes[3].set_ylim(0-lim, 1+lim)
    
    axes[4].plot(eps,0.5,'bo', markersize=5, color='red')
    axes[4].plot(x, sigmoid_f(x, coefs[1], eps), color='red')
    axes[4].set_title(f'Sigmoid Function with coef={coefs2[1]}')
    axes[4].set_xlim(0-lim, 1+lim)
    axes[4].set_ylim(0-lim, 1+lim)
    


    # 레이아웃 조정
    plt.tight_layout()

    # 그래프 표시
    plt.savefig("./figures/all.png")