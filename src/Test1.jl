module Test1
export S
struct S 
   a
   b
end

(s::S)(x) = (s.a + s.b) * x 

end