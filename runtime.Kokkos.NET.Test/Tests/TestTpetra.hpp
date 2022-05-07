    bool success;

    Teuchos::OSTab tab0(std::cout);

    std::cout << "Make sure that taking a subview of a Kokkos::DualView "
                 "with zero rows and nonzero columns produces a Kokkos::DualView "
                 "with the correct number of columns."
              << endl;

    Teuchos::OSTab tab1(std::cout);

    auto comm = getDefaultComm();

    // Creating a Map instance takes care of Kokkos initialization and
    // finalization automatically.
    Tpetra::Map<> map(comm->getSize(), 1, 0, comm);

    if(!Kokkos::is_initialized())
    {
        return; // avoid crashes if initialization failed
    }

    std::cout << "Successfully initialized execution space, if necessary" << endl;

    size_type                 newNumRows = 0;
    size_type                 newNumCols = 0;
    std::pair<size_t, size_t> rowRng(0, 0);
    std::pair<size_t, size_t> colRng(0, 0);
    dual_view_type            X_sub;

    std::cout << "Make sure that Tpetra::MultiVector::dual_view_type has rank 2" << endl;

    TEST_EQUALITY_CONST((int)dual_view_type::rank, 2);

    size_type numRows = 0;
    size_type numCols = 10;
    std::cout << "Create a " << numRows << " x " << numCols << " DualView" << endl;
    dual_view_type X("X", numRows, numCols);

    TEST_EQUALITY_CONST(X.extent(0), numRows);
    TEST_EQUALITY_CONST(X.extent(1), numCols);
    TEST_EQUALITY_CONST(X.d_view.extent(0), numRows);
    TEST_EQUALITY_CONST(X.d_view.extent(1), numCols);
    TEST_EQUALITY_CONST(X.h_view.extent(0), numRows);
    TEST_EQUALITY_CONST(X.h_view.extent(1), numCols);

    std::cout << endl;

    newNumRows = numRows;
    newNumCols = 5;
    colRng     = std::pair<size_t, size_t>(0, newNumCols);

    std::cout << "Create a " << newNumRows << " x " << newNumCols << " subview using (ALL, pair(" << colRng.first << "," << colRng.second << "))" << endl;

    X_sub = subview(X, ALL(), colRng);

    std::cout << "X_sub claims to be " << X_sub.extent(0) << " x " << X_sub.extent(1) << endl;

    TEST_EQUALITY_CONST(X_sub.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(1), newNumCols);

    std::cout << endl;

    newNumRows = numRows;
    newNumCols = 1;
    colRng     = std::pair<size_t, size_t>(0, newNumCols);

    std::cout << "Create a " << newNumRows << " x " << newNumCols << " subview using (ALL, pair(" << colRng.first << "," << colRng.second << "))" << endl;

    X_sub = subview(X, ALL(), colRng);

    std::cout << "X_sub claims to be " << X_sub.extent(0) << " x " << X_sub.extent(1) << endl;

    TEST_EQUALITY_CONST(X_sub.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(1), newNumCols);
    std::cout << endl;

    newNumRows = 0;
    newNumCols = numCols;
    rowRng     = std::pair<size_t, size_t>(0, 0);

    std::cout << "Create a " << newNumRows << " x " << newNumCols << " subview using (pair(" << rowRng.first << "," << rowRng.second << "), ALL)" << endl;

    X_sub = subview(X, rowRng, ALL());

    std::cout << "X_sub claims to be " << X_sub.extent(0) << " x " << X_sub.extent(1) << endl;

    TEST_EQUALITY_CONST(X_sub.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(1), newNumCols);

    std::cout << endl;

    newNumRows = 0;
    newNumCols = 5;
    rowRng     = std::pair<size_t, size_t>(0, newNumRows);
    colRng     = std::pair<size_t, size_t>(0, newNumCols);

    std::cout << "Create a " << newNumRows << " x " << newNumCols << " subview using (pair(" << rowRng.first << "," << rowRng.second << "), pair(" << colRng.first << "," << colRng.second << "))" << endl;

    X_sub = subview(X, rowRng, colRng);

    std::cout << "X_sub claims to be " << X_sub.extent(0) << " x " << X_sub.extent(1) << endl;

    TEST_EQUALITY_CONST(X_sub.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.d_view.extent(1), newNumCols);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(0), newNumRows);
    TEST_EQUALITY_CONST(X_sub.h_view.extent(1), newNumCols);

    std::cout << endl;