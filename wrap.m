function [nlml, a, c] = wrap(p, data, qx);
p.qx=qx;
[nlml, a, c] = vgpt(p,data);
a=rmfield(a,'qx');
