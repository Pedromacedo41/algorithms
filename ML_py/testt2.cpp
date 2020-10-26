#include <bits/stdc++.h>
using namespace std;

#define st first
#define nd second
#define mp make_pair
#define pb push_back
#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i,n) for(int i=0; i < n; i++)


typedef long long ll;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<int, pii> piii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;

vector<string> rep;

int main(){
    int testcases;
    cin >> testcases;
    f(i,testcases){
        int x,y;
        cin >> x >> y;
        if(x==1){
            if(y!=1) rep.pb("NO");
            else rep.pb("YES");
        }
        else{
            if(y<=x) rep.pb("YES");
            else{
                if(x>=4) rep.pb("YES");
                else{
                    int d = x;
                    while(true){
                        if(d%2==0){
                            d=3*d/2;
                            if(d>=y) break;
                        }else{
                            break;
                        }
                    }
                    if(d>=y) rep.pb("YES");
                    else rep.pb("NO");
                }
            }
        }
    }
    f(i,testcases){
        cout << rep[i] << endl;
    }
    return 0;
}