module tools

export print_mpi

function print_mpi( text, rank, root=0 )
  if rank == root
    print( text )
  end
end


end