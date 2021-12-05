import torch
a = torch.rand(3, 4)
#a = a.unsqueeze(0)
#print(a.reshape(3,4,1))
b = torch.rand(3, 4)
#b = b.unsqueeze(0)
print(b)
c = torch.stack([a, b, b, b, b], dim=1)


c = torch.rand(3, 20)
print(c)
c = c.reshape(3, 5, 4)
print(c.shape)

d = torch.rand(3, 5)
d = d.reshape(3,5,1)
print(d)

e = c*d
print(c*d)

print(torch.mean(e, axis=1))
print(torch.mean(e, axis=1).reshape(6,2))

f = torch.mean(e, axis=1).reshape(6,2)
print(f)
#f = f.reshape(f.shape[0],1,f.shape[1])
#print(f)

f = torch.stack([f,f,f], dim=1)
print(f)

f = f.reshape(f.shape[0]*f.shape[1], f.shape[2])
print(f)


"""
a = torch.rand(1, 3, 4)
print(a.shape)
b = torch.rand(3, 4)
print(b.shape)
b = b.unsqueeze(0)
print(b.shape)
c = torch.cat([a, b], dim=0)
print(c.shape)
"""