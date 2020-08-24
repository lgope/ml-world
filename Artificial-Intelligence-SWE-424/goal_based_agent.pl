edge(a,b).
edge(b,c).
edge(c,d).
edge(d,e).
edge(d,f).
edge(f,g).
edge(g,h).
edge(h,i).
edge(g,j).
edge(j,k).
edge(k,l).
edge(k,m).
edge(m,n).
edge(m,o).
edge(o,p).

path(X,Y):-edge(X,Y).
path(X,Y):-edge(X,Z),path(Z,Y),write(Z).