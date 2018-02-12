function[idx] = createregion(vector_size)
   N = sum(vector_size);
   d = size(vector_size,1);
   idx = zeros(N,1);
   
   begin = 1;
   region = 1;
   for j = 1:d
      tmp = vector_size(j,1);
      idx(begin:(begin+tmp-1),1) = region; 
      region = region + 1;
      begin = begin + tmp;
   end
   
end