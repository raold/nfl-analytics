#!/usr/bin/env python3
"""
Database Connection Pool Validation Script

Tests connection pool configuration for A/B test production deployment.
Validates that the pool can handle concurrent connections without leaks.

Created: November 14, 2025
"""

import sys
import time
from contextlib import contextmanager

sys.path.append("/Users/dro/rice/nfl-analytics")

import psycopg2
from psycopg2.pool import SimpleConnectionPool

# Database configuration (matches real_time_api.py)
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Pool configuration
MIN_CONNECTIONS = 1
MAX_CONNECTIONS = 20


@contextmanager
def get_connection(pool):
    """Context manager for connection from pool"""
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def test_basic_connection():
    """Test 1: Basic connection creation and query"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Connection")
    print("=" * 70)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ Connection successful")
        print(f"   PostgreSQL version: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_connection_pool():
    """Test 2: Connection pool creation and basic operation"""
    print("\n" + "=" * 70)
    print("TEST 2: Connection Pool Creation")
    print("=" * 70)

    try:
        pool = SimpleConnectionPool(MIN_CONNECTIONS, MAX_CONNECTIONS, **DB_CONFIG)
        print(f"✅ Pool created (min={MIN_CONNECTIONS}, max={MAX_CONNECTIONS})")

        # Get a connection
        with get_connection(pool) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()[0]
            cursor.close()
            assert result == 1
            print(f"✅ Connection from pool works")

        pool.closeall()
        print(f"✅ Pool closed successfully")
        return True
    except Exception as e:
        print(f"❌ Pool test failed: {e}")
        return False


def test_concurrent_connections():
    """Test 3: Multiple concurrent connections"""
    print("\n" + "=" * 70)
    print("TEST 3: Concurrent Connections (10 simultaneous)")
    print("=" * 70)

    try:
        pool = SimpleConnectionPool(MIN_CONNECTIONS, MAX_CONNECTIONS, **DB_CONFIG)

        # Get 10 connections simultaneously
        connections = []
        for i in range(10):
            conn = pool.getconn()
            connections.append(conn)

        print(f"✅ Acquired 10 connections from pool")

        # Execute query on each
        for i, conn in enumerate(connections, 1):
            cursor = conn.cursor()
            cursor.execute("SELECT model_id FROM ml.model_registry LIMIT 1;")
            cursor.close()

        print(f"✅ Executed queries on all 10 connections")

        # Return all connections
        for conn in connections:
            pool.putconn(conn)

        print(f"✅ Returned all connections to pool")

        pool.closeall()
        return True
    except Exception as e:
        print(f"❌ Concurrent connection test failed: {e}")
        return False


def test_connection_leak():
    """Test 4: Connection leak detection"""
    print("\n" + "=" * 70)
    print("TEST 4: Connection Leak Detection")
    print("=" * 70)

    try:
        pool = SimpleConnectionPool(MIN_CONNECTIONS, MAX_CONNECTIONS, **DB_CONFIG)

        # Get max connections
        connections = []
        for i in range(MAX_CONNECTIONS):
            conn = pool.getconn()
            connections.append(conn)

        print(f"✅ Acquired {MAX_CONNECTIONS} connections (pool exhausted)")

        # Try to get one more (should raise PoolError)
        try:
            pool.getconn()
            print(f"❌ ERROR: Pool allowed more than max connections!")
            pool.closeall()
            return False
        except Exception as e:
            print(f"✅ Pool correctly rejected connection #{MAX_CONNECTIONS + 1}")

        # Return all connections
        for conn in connections:
            pool.putconn(conn)

        print(f"✅ Returned all connections (no leak detected)")

        pool.closeall()
        return True
    except Exception as e:
        print(f"❌ Connection leak test failed: {e}")
        return False


def test_query_performance():
    """Test 5: Query performance under load"""
    print("\n" + "=" * 70)
    print("TEST 5: Query Performance (100 queries)")
    print("=" * 70)

    try:
        pool = SimpleConnectionPool(MIN_CONNECTIONS, MAX_CONNECTIONS, **DB_CONFIG)

        # Run 100 queries and measure time
        start_time = time.time()
        for i in range(100):
            with get_connection(pool) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM ml.predictions
                    WHERE model_id IN (38, 39);
                    """
                )
                cursor.close()

        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / 100) * 1000

        print(f"✅ Completed 100 queries in {elapsed:.2f}s")
        print(f"   Average query time: {avg_time_ms:.2f}ms")

        if avg_time_ms < 10:
            print(f"✅ Performance: Excellent (<10ms avg)")
        elif avg_time_ms < 50:
            print(f"✅ Performance: Good (<50ms avg)")
        elif avg_time_ms < 100:
            print(f"⚠️  Performance: Acceptable (<100ms avg)")
        else:
            print(f"❌ Performance: Poor (>{100}ms avg)")
            pool.closeall()
            return False

        pool.closeall()
        return True
    except Exception as e:
        print(f"❌ Query performance test failed: {e}")
        return False


def test_ab_test_queries():
    """Test 6: A/B test monitoring queries"""
    print("\n" + "=" * 70)
    print("TEST 6: A/B Test Monitoring Queries")
    print("=" * 70)

    try:
        pool = SimpleConnectionPool(MIN_CONNECTIONS, MAX_CONNECTIONS, **DB_CONFIG)

        # Test Panel 1 query
        with get_connection(pool) as conn:
            cursor = conn.cursor()
            start_time = time.time()
            cursor.execute(
                """
                SELECT
                    m.model_name,
                    COUNT(p.prediction_id) as total_predictions
                FROM ml.model_registry m
                LEFT JOIN ml.predictions p ON m.model_id = p.model_id
                WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
                GROUP BY m.model_id, m.model_name
                ORDER BY m.model_name;
                """
            )
            results = cursor.fetchall()
            elapsed_ms = (time.time() - start_time) * 1000
            cursor.close()

        print(f"✅ Panel 1 query executed in {elapsed_ms:.2f}ms")
        print(f"   Results: {len(results)} models")

        if elapsed_ms < 1000:
            print(f"✅ Query performance: Excellent (<1s)")
        else:
            print(f"❌ Query performance: Too slow (>{1}s)")
            pool.closeall()
            return False

        pool.closeall()
        return True
    except Exception as e:
        print(f"❌ A/B test query failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("\n" + "=" * 70)
    print("DATABASE CONNECTION POOL VALIDATION")
    print("=" * 70)
    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"Pool config: min={MIN_CONNECTIONS}, max={MAX_CONNECTIONS}")

    tests = [
        test_basic_connection,
        test_connection_pool,
        test_concurrent_connections,
        test_connection_leak,
        test_query_performance,
        test_ab_test_queries,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - Test {i}: {test.__doc__.split(':')[1].strip()}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ GO FOR PRODUCTION DEPLOYMENT")
    else:
        print("❌ NO-GO - Fix failing tests before deployment")

    print("=" * 70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
