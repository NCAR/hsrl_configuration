function file_list = trace_file_tree(base_path,file_str,filesize)
% base_path - directory to start recursive search for files matching
% file_str
% file_str - string used to match files.  right now it has to be the end of
% the file
% filesize - minimum file size (in bytes) to include it in the tree

pathList = dir(base_path);
file_list = {};
for ai = 1:length(pathList)
    if ~ strcmp(pathList(ai).name,'.') && ~ strcmp(pathList(ai).name,'..')
        if pathList(ai).isdir
            % if the object is a directory, recursively enter it and pull
            % the files from it
            new_file_list = trace_file_tree([pathList(ai).folder,'/',pathList(ai).name],file_str,filesize);
            file_list = [file_list, new_file_list];
        elseif endsWith(pathList(ai).name,file_str) && pathList(ai).bytes > filesize
            % if the object is a netcdf file, concatenate it to the file
            % list
            file_list = [file_list, [pathList(ai).folder,'/',pathList(ai).name]];
        end
    end
end