Hashing
=====

### Hashability in Python
Hashability makes an object <b>usable as a dictionary key and a set member, because these data structures use the hash value internally.</b>
Python immutable built-in objects are hashable; mutable containers (such as lists or dictionaries) are not.
Objects which are instances of user-defined classes are hashable by default.
They all compare unequal (except with themselves), and their hash value is derived from their id().

[출처](http://zetcode.com/python/hashing/)
---


