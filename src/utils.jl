"""
    onehot_encode(y, labels)
    
Function to one hot encode `y` as per the labels provided as `1:n` for n labels. 
"""
function onehot_encode(y, labels)
   onehot_arr = zeros(length(labels), length(y))
   for i in 1:length(y)
       onehot_arr[Int(y[i]) + 1, i] = 1
   end
   onehot_arr
end

# Onecold
function onecold(y)
    onecold_arr = zeros(size(y,2))
    argmax_vec = argmax(y, dims = 1)
    for i in 1:size(y,2)
        onecold_arr[i] = argmax_vec[i].I[1]
    end
    return onecold_arr
end
